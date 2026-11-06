"""Agnes 图像/视频生成：多 Key 池、异步视频任务、结果直接入库。"""

from __future__ import annotations

import base64
import binascii
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from app.config import (
    AGNES_IMAGE_MODEL,
    AGNES_MEDIA_BASE_URL,
    AGNES_VIDEO_CREATE_TIMEOUT,
    AGNES_VIDEO_MODEL,
    AGNES_VIDEO_QUERY_URL,
    COVERS_DIR,
    DATA_DIR,
    PUBLIC_BASE_URL,
    VIDEOS_DIR,
)
from app.db import add_video
from app.services.ai_pool import AI_POOL, AIPoolError
from app.services.media import make_cover, probe_video
from app.services.tasks import task_manager

GENERATED_IMAGE_DIR = DATA_DIR / "generated" / "images"
GENERATED_VIDEO_DIR = DATA_DIR / "generated" / "videos"
GENERATED_REFERENCE_DIR = DATA_DIR / "generated" / "references"


class MediaAIError(RuntimeError):
    pass


def _profile_context(
    prompt: str,
    character_id: str = "",
    scene_id: str = "",
    style_id: str = "",
    transition: str = "",
) -> tuple[str, list[str], dict[str, Any] | None]:
    from app.services.profiles import resolve_bundle

    bundle = resolve_bundle(character_id, scene_id, style_id)
    character = bundle.get("character")
    if not character and re.search(r"人物|男人|女人|老人|女孩|男孩|角色|陈默|回声", prompt):
        raise MediaAIError("含人物的生成必须先选择人物档案，避免每次生成换一张脸")
    sections = [bundle.get("prompt") or "", prompt.strip()]
    if transition == "morph":
        sections.append(
            "关键转场：同一人物在同一机位一对一缓慢 morph，只改变档案年龄规则允许变化的部位；"
            "服装、痣、发型、背景摆设和光线保持连续，禁止叠化出第二个人影或更换场景。"
        )
    references = [item.get("data_uri") or item.get("url") for item in bundle.get("references") or []]
    return "\n".join(part for part in sections if part), [item for item in references if item], character


def _image_consistency(path: Path, character: dict[str, Any] | None) -> dict[str, Any]:
    from app.config import VISION_MODEL

    if not character:
        return {"status": "skipped", "reason": "未使用人物档案"}
    if not VISION_MODEL:
        return {"status": "unknown", "reason": "未配置视觉模型，无法自动比对人物一致性"}
    try:
        from app.services.understanding import _request_json

        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "对比人物档案参考图和新生成图。检查是否同一人、固定面部特征、服装和年龄规则。"
                "只输出 JSON：{\"score\":0,\"passed\":false,\"mismatches\":[],\"reason\":\"\"}。"
                "80 分及以上才算通过。\n人物档案：" + str({
                    key: character.get(key) for key in ("name", "fixed_features", "clothing", "age_rules")
                })
            ),
        }]
        for reference in character.get("references") or []:
            data_uri = reference.get("data_uri")
            if data_uri:
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        generated = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": "下面是新生成图"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{generated}"}})
        result = _request_json([{"role": "user", "content": content}], vision=True)
        score = max(0, min(100, float(result.get("score") or 0)))
        return {
            "status": "pass" if score >= 80 and result.get("passed", True) else "review",
            "score": score,
            "mismatches": result.get("mismatches") or [],
            "reason": result.get("reason") or "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "reason": f"一致性比对失败：{exc}"[:300]}


def _is_public_url(url: str) -> bool:
    """判断一个 URL 的主机是不是公网地址。localhost / 127.* / 10.* /
    172.16-31.* / 192.168.* / ::1 都不是公网，Agnes 从它自己的服务器
    上拉图时必然失败——此前代码静默发这种地址，失败原因很难定位。"""
    import ipaddress
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").strip().lower()
    except ValueError:
        return False
    if not host or host in {"localhost"}:
        return False
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_global
    except ValueError:
        # 不是 IP 字面量：只要不是 localhost 就当作域名，交给 Agnes 侧解析；
        # 内网域名不在此列，部署时仍应以公网域名为准。
        return True


def _validate_reference_values(values: list[str]) -> None:
    """参考图/首尾帧校验：data_uri 直接放行；http(s) 地址若是内网/
    回环地址，直接报明确错误，不要发给 Agnes 后再看它失败。"""
    for value in values:
        value = (value or "").strip()
        if not value or value.startswith("data:image/"):
            continue
        if value.startswith("http://") or value.startswith("https://"):
            if not _is_public_url(value):
                raise MediaAIError(
                    "参考图地址是内网/回环地址，Agnes 无法从它的服务器上访问："
                    f"{value[:80]}。请把 VIDEO_HUB_PUBLIC_BASE_URL 改成公网地址，"
                    "或改用新上传的参考图（会优先发 data_uri）。"
                )


def _download(url: str, target: Path, timeout: float = 180) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for chunk in response.iter_content(1024 * 512):
            if chunk:
                handle.write(chunk)
    if not target.is_file() or target.stat().st_size <= 0:
        raise MediaAIError("下载生成结果失败")


def save_reference(data: bytes, filename: str) -> dict[str, Any]:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    GENERATED_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"ref_{uuid.uuid4().hex}{suffix}"
    target = GENERATED_REFERENCE_DIR / name
    target.write_bytes(data)
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix[1:]}"
    public_ok = _is_public_url(PUBLIC_BASE_URL)
    return {
        # 公网地址不可用时 url 置空，绝不发一个必定失败的 127.0.0.1 地址；
        # 前端应优先用 data_uri（见 AIMediaPage 的 refValue）。
        "url": f"{PUBLIC_BASE_URL}/api/ai/media/references/{name}" if public_ok else "",
        "url_usable": public_ok,
        "url_warning": (
            None if public_ok
            else "PUBLIC_BASE_URL 不是公网地址（当前是内网/回环地址），"
                 "url 字段已置空；请把 VIDEO_HUB_PUBLIC_BASE_URL 改成公网地址，"
                 "或直接用本次返回的 data_uri 作为参考图。"
        ),
        "data_uri": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        "name": name,
        "size": len(data),
    }


def _require_keys() -> None:
    if not AI_POOL.has_keys():
        raise MediaAIError("没有配置 AI API Key")


def generate_image(
    prompt: str,
    size: str = "2K",
    ratio: str = "16:9",
    images: list[str] | None = None,
    character_profile_id: str = "",
    scene_profile_id: str = "",
    style_profile_id: str = "",
) -> dict[str, Any]:
    _require_keys()
    prompt, profile_refs, character = _profile_context(prompt, character_profile_id, scene_profile_id, style_profile_id)
    all_images = [*profile_refs, *(images or [])][:5]
    result: dict[str, Any] = {}
    for attempt in range(1, 3):
        result = _generate_image_once(prompt, size, ratio, all_images)
        consistency = _image_consistency(GENERATED_IMAGE_DIR / result["name"], character)
        result["consistency"] = consistency
        result["attempts"] = attempt
        if consistency.get("status") != "review":
            break
    return result


def _generate_image_once(prompt: str, size: str, ratio: str, images: list[str]) -> dict[str, Any]:
    if not prompt.strip():
        raise MediaAIError("提示词不能为空")
    _validate_reference_values(images)
    payload: dict[str, Any] = {
        "model": AGNES_IMAGE_MODEL,
        "prompt": prompt,
        "size": size if size in {"1K", "2K", "3K", "4K"} else "2K",
        "ratio": ratio,
        "extra_body": {"response_format": "url"},
    }
    if images:
        payload["image"] = images
    try:
        response = AI_POOL.request(
            "POST",
            f"{AGNES_MEDIA_BASE_URL}/images/generations",
            json_body=payload,
            timeout=240,
        )
    except AIPoolError as exc:
        raise MediaAIError(f"图像接口不可用：{exc}") from exc
    if response.status_code != 200:
        raise MediaAIError(f"图像接口错误 {response.status_code}：{response.text[:300]}")
    data = response.json()
    item = (data.get("data") or [{}])[0]
    GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"ai_{uuid.uuid4().hex}.png"
    target = GENERATED_IMAGE_DIR / name
    if item.get("url"):
        _download(str(item["url"]), target)
    elif item.get("b64_json"):
        try:
            target.write_bytes(base64.b64decode(item["b64_json"]))
        except (ValueError, binascii.Error) as exc:
            raise MediaAIError("图像 Base64 解码失败") from exc
    else:
        raise MediaAIError("图像接口没有返回图片")
    return {
        "ok": True,
        "url": f"/api/ai/media/images/{name}",
        "name": name,
        "model": AGNES_IMAGE_MODEL,
        "size": target.stat().st_size,
    }


def _create_video(
    prompt: str,
    seconds: str,
    aspect_ratio: str,
    task,
    first_frame: str = "",
    last_frame: str = "",
    images: list[str] | None = None,
) -> dict[str, Any]:
    images = [item for item in (images or []) if item][:5]
    # 参考图地址先校验：内网地址发给 Agnes 必然失败，不要等排队超时才暴露。
    _validate_reference_values(
        [v for v in ([first_frame, last_frame] + images) if v]
    )
    if first_frame or last_frame:
        mode = "keyframe"
    elif images:
        mode = "reference"
    else:
        mode = "text"
    payload = {
        "model": AGNES_VIDEO_MODEL,
        "prompt": prompt,
        "seconds": seconds,
        "mode": mode,
        "size": "720P",
        "aspect_ratio": aspect_ratio,
    }
    if mode == "keyframe":
        if first_frame:
            payload["first_frame"] = first_frame
        if last_frame:
            payload["last_frame"] = last_frame
    elif mode == "reference":
        payload["images"] = images
    last_error = ""
    attempt = 0
    # 总时限：此前是 while not cancelled 无限重试，队列满时可无限挂起。
    # 超过 AGNES_VIDEO_CREATE_TIMEOUT（默认 1800 秒）明确失败，错误里带
    # 最后一次错误，调用方（_generate_video_task）不再对外层重试 3 次。
    # 有效时限至少 60 秒，避免配置过小导致一次重试都等不到；
    # 以下所有日志/报错都用同一个 effective_timeout，避免“写 5 秒、实际 60 秒”。
    effective_timeout = max(60, int(AGNES_VIDEO_CREATE_TIMEOUT or 1800))
    deadline = time.monotonic() + effective_timeout
    while not task.cancelled:
        if time.monotonic() >= deadline:
            raise MediaAIError(
                f"视频创建排队超时（已等待 {effective_timeout} 秒）。"
                f"最后一次错误：{last_error or '无'}"
            )
        attempt += 1
        try:
            response = AI_POOL.request(
                "POST",
                f"{AGNES_MEDIA_BASE_URL}/videos",
                json_body=payload,
                timeout=90,
                busy_markers=("video_queue_full",),
            )
        except AIPoolError as exc:
            last_error = str(exc)
            task.log(f"视频接口暂不可用：{last_error[:120]}，30 秒后换 Key 再试")
            time.sleep(min(30, max(0, deadline - time.monotonic())))
            continue
        if response.status_code == 200:
            data = response.json()
            video_id = data.get("video_id") or data.get("id")
            if video_id:
                data["_key"] = AI_POOL.last_key
                return data
            last_error = "视频接口没有返回 video_id"
            task.log(last_error + "，继续重试")
            time.sleep(min(15, max(0, deadline - time.monotonic())))
            continue
        last_error = f"{response.status_code} {response.text[:300]}"
        if "queue_full" in last_error.lower() or "队列已满" in last_error:
            elapsed = int(time.monotonic() - (deadline - effective_timeout))
            if attempt == 1 or attempt % 5 == 0:
                task.log(
                    f"视频队列已满，已排队约 {elapsed} 秒，继续排队"
                    f"（第 {attempt} 次，总时限 {effective_timeout} 秒）"
                )
            time.sleep(min(60, 15 + attempt * 5, max(0, deadline - time.monotonic())))
            continue
        if response.status_code in (400, 401, 403, 404, 422):
            raise MediaAIError(last_error)
        task.log(f"视频接口 HTTP {response.status_code}，30 秒后重试")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise MediaAIError("视频任务已取消")


def _poll_video(video_id: str, task, task_key: str = "", timeout: float = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    params = {"video_id": video_id, "model_name": AGNES_VIDEO_MODEL}
    while time.monotonic() < deadline:
        if task.cancelled:
            raise MediaAIError("视频任务已取消")
        key = task_key or AI_POOL.acquire()
        if not key:
            time.sleep(5)
            continue
        try:
            response = requests.get(
                AGNES_VIDEO_QUERY_URL,
                params=params,
                headers={"Authorization": f"Bearer {key}"},
                timeout=60,
            )
        except requests.RequestException as exc:
            task.log(f"视频查询网络错误：{exc}")
            time.sleep(5)
            continue
        if response.status_code == 429:
            AI_POOL.mark_failure(key, 429, response.text[:160])
            time.sleep(5)
            continue
        if response.status_code == 404:
            raise MediaAIError("视频任务不存在或当前 Key 无权访问")
        if response.status_code != 200:
            AI_POOL.mark_failure(key, response.status_code, response.text[:160])
            task.log(f"视频查询 HTTP {response.status_code}")
            time.sleep(10)
            continue
        AI_POOL.mark_success(key)
        data = response.json()
        status = str(data.get("status") or "")
        progress = float(data.get("progress") or 0)
        task.progress = min(90.0, max(5.0, progress * 0.9))
        task.log(f"视频生成中：{status or 'pending'} {progress:.0f}%")
        if status == "completed":
            return data
        if status == "failed":
            raise MediaAIError(str(data.get("error") or "视频生成失败"))
        time.sleep(5)
    raise MediaAIError("视频生成超时")


def _video_consistency(path: Path, character: dict[str, Any] | None) -> dict[str, Any]:
    if not character:
        return {"status": "skipped", "reason": "未使用人物档案"}
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="consistency-", dir=DATA_DIR / "tmp") as temp:
        frame = Path(temp) / "frame.jpg"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", "0.5", "-i", str(path),
             "-frames:v", "1", "-q:v", "3", str(frame)],
            capture_output=True, timeout=120, check=False,
        )
        if result.returncode != 0 or not frame.is_file():
            return {"status": "unknown", "reason": "无法抽取生成视频的关键帧"}
        return _image_consistency(frame, character)


def _generate_video_task(
    task,
    prompt: str,
    seconds: str,
    aspect_ratio: str,
    first_frame: str = "",
    last_frame: str = "",
    images: list[str] | None = None,
    save_to_library: bool = True,
    character: dict[str, Any] | None = None,
) -> None:
    task.status = "running"
    mode = "keyframe" if (first_frame or last_frame) else ("reference" if images else "text")
    task.log(f"提交 {AGNES_VIDEO_MODEL}：{seconds}s {aspect_ratio} · {mode}")
    error_chain: list[str] = []

    for attempt in range(1, 4):
        if task.cancelled:
            task.log("任务已取消")
            return
        if attempt > 1:
            task.log(f"第 {attempt}/3 次生成，自动换 Key 重试…")
            task.progress = 0
        try:
            created = _create_video(prompt, seconds, aspect_ratio, task, first_frame, last_frame, images)
            video_id = str(created.get("video_id") or created.get("id"))
            task_key = str(created.get("_key") or "")
            task.log(f"视频任务已创建：{video_id}")
            data = _poll_video(video_id, task, task_key)
            url = data.get("url")
            if not url:
                raise MediaAIError("视频任务完成但没有下载地址")
            GENERATED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            temp = GENERATED_VIDEO_DIR / f"ai_{uuid.uuid4().hex}.mp4"
            task.log("正在下载生成视频…")
            _download(str(url), temp, timeout=600)
            consistency = _video_consistency(temp, character)
            if consistency.get("status") == "review" and attempt == 1:
                temp.unlink(missing_ok=True)
                task.log("人物/场景一致性未通过，自动重生成一次")
                continue
            if consistency.get("status") == "review":
                save_to_library = False
            meta = probe_video(temp)
            title = (prompt[:60] or "AI 生成视频").strip()
            if save_to_library:
                final = VIDEOS_DIR / f"ai_{uuid.uuid4().hex}.mp4"
                shutil.move(str(temp), final)
                cover_name = final.stem + ".jpg"
                cover = cover_name if make_cover(final, COVERS_DIR / cover_name) else None
                video = add_video(
                    title=title,
                    filename=final.name,
                    size=final.stat().st_size,
                    duration=meta.get("duration"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    cover=cover,
                    source_platform="ai_generated",
                    source_url=url,
                )
                task.result = {
                    "video": video,
                    "video_url": f"/api/videos/{video['id']}/download",
                    "saved_to_library": True,
                    "model": AGNES_VIDEO_MODEL,
                }
            else:
                name = f"ai_{uuid.uuid4().hex}.mp4"
                final = GENERATED_VIDEO_DIR / name
                shutil.move(str(temp), final)
                task.result = {
                    "video": None,
                    "video_url": f"/api/ai/media/videos/{name}",
                    "saved_to_library": False,
                    "title": title,
                    "duration": meta.get("duration"),
                    "size": final.stat().st_size,
                    "model": AGNES_VIDEO_MODEL,
                }
            task.result["consistency"] = consistency
            task.log("视频已生成" + ("并加入素材库" if save_to_library else "（未加入素材库）"))
            task.progress = 100.0
            task.finish("partial" if consistency.get("status") == "review" else "success")
            return
        except Exception as exc:  # noqa: BLE001
            if task.cancelled:
                task.log("任务已取消")
                return
            error_chain.append(f"第 {attempt} 次：{exc}")
            # 排队超时是确定性失败（已等满总时限），外层再重试 3 次只会
            # 把总等待拉到 3 倍，没有意义，直接失败并保留完整错误链。
            if "排队超时" in str(exc):
                task.error = {
                    "code": "ai_video_timeout",
                    "message": str(exc)[:300],
                    "action": "生成队列已满或等待过久。稍后重试，或先确认 Key 池里有可用密钥。",
                }
                task.result = {"error": task.error["message"], "error_chain": error_chain}
                task.log(f"AI 视频生成失败（排队超时，不再重试）：{exc}")
                task.finish("failed")
                return
            if attempt < 3:
                task.log(f"第 {attempt}/3 次生成失败：{str(exc)[:180]}，10 秒后换 Key 重试")
                time.sleep(10)
                continue
            task.error = {
                "code": "ai_keys_missing" if "没有配置 AI API Key" in str(exc) else "ai_video_failed",
                "message": str(exc)[:300],
                "action": "在服务器配置 VIDEO_HUB_AI_API_KEY 或 AGNES_API_KEY 后重启后端。" if "没有配置 AI API Key" in str(exc) else "查看任务日志里的失败原因，改描述或参考图后再试。",
            }
            task.result = {"error": task.error["message"], "error_chain": error_chain}
            task.log(f"AI 视频生成失败（已重试 3 次）：{exc}")
            task.finish("failed")


def start_video_generation(
    prompt: str,
    seconds: int = 5,
    aspect_ratio: str = "16:9",
    first_frame: str = "",
    last_frame: str = "",
    images: list[str] | None = None,
    save_to_library: bool = True,
    character_profile_id: str = "",
    scene_profile_id: str = "",
    style_profile_id: str = "",
    transition: str = "",
) -> dict[str, Any]:
    _require_keys()
    if transition == "morph" and (not first_frame or not last_frame):
        raise MediaAIError("一对一 morph 转场必须同时提供首帧和尾帧")
    prompt, profile_refs, character = _profile_context(
        prompt, character_profile_id, scene_profile_id, style_profile_id, transition
    )
    images = [*(profile_refs if not (first_frame or last_frame) else []), *(images or [])][:5]
    _validate_reference_values([v for v in ([first_frame, last_frame] + list(images or [])) if v])
    seconds = max(4, min(int(seconds or 5), 12))
    if aspect_ratio not in {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}:
        aspect_ratio = "16:9"
    mode = "keyframe" if (first_frame or last_frame) else ("reference" if images else "text")
    task = task_manager.create("ai", f"AI 视频 · {prompt[:30]}")
    threading.Thread(
        target=_generate_video_task,
        args=(task, prompt, str(seconds), aspect_ratio, first_frame, last_frame, images, save_to_library, character),
        daemon=True,
    ).start()
    return {"task_id": task.id, "mode": mode}


def start_video_batch(shots: list[dict[str, Any]], shared: dict[str, Any]) -> dict[str, Any]:
    if not shots or len(shots) > 10:
        raise MediaAIError("连续镜头批量生成需要 1-10 个镜头")
    tasks = []
    for index, shot in enumerate(shots):
        prompt = str(shot.get("prompt") or "").strip()
        if not prompt:
            raise MediaAIError(f"第 {index + 1} 个镜头缺少描述")
        result = start_video_generation(
            prompt,
            seconds=int(shot.get("seconds") or 5),
            aspect_ratio=str(shot.get("aspect_ratio") or shared.get("aspect_ratio") or "16:9"),
            first_frame=str(shot.get("first_frame") or ""),
            last_frame=str(shot.get("last_frame") or ""),
            save_to_library=bool(shot.get("save_to_library", True)),
            character_profile_id=str(shared.get("character_profile_id") or ""),
            scene_profile_id=str(shared.get("scene_profile_id") or ""),
            style_profile_id=str(shared.get("style_profile_id") or ""),
            transition=str(shot.get("transition") or ""),
        )
        tasks.append(result)
    return {"tasks": tasks, "count": len(tasks)}
