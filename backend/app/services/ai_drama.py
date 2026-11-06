"""AI 短剧流水线：小说 → 分集分镜 → 角色锚定图 → 逐镜首帧 → 图生视频 → 合成单集。

复用现有能力：
- app.services.understanding._request_json：LLM JSON 输出（AI_POOL 多 Key 轮换）
- app.services.ai_media：Agnes 生图 / 生视频（多 Key 池、失败换 Key）
- app.services.profiles：人物档案（固定层提示词 + 锚定参考图，锁角色一致性）

失败策略：每个生成步骤失败后自动换 Key 重试，直到成功；为防止 Key 全部失效
导致死循环，单步骤默认最多重试 max_attempts 次（可在创建项目时调整），超限后
项目标记 failed，产物全部落盘，随时可「继续」从断点续跑。
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import DATA_DIR
from app.services import ai_media, profiles
from app.services.ai_pool import AI_POOL
from app.services.media import probe_video
from app.services.tasks import task_manager

DRAMA_DIR = DATA_DIR / "drama"
STYLES = {
    "realistic": "写实风格，电影级光影，高清细节，浅景深",
    "semi": "半写实风格，精致插画质感，柔和电影光",
    "anime": "日系动漫风格，精致赛璐璐上色，电影感构图",
    "guofeng": "国风工笔风格，东方美学，水墨意境，电影级光影",
}
THREADS: dict[str, threading.Thread] = {}
_LOCK = threading.RLock()


class DramaError(RuntimeError):
    """确定性错误（取消、数据缺失），不做无意义重试。"""


class _RetryableError(RuntimeError):
    """生成失败：换 Key 重试直到成功。"""


# ---------------------------------------------------------------- 项目持久化

def _dir(project_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", project_id or ""):
        raise DramaError("项目不存在")
    return DRAMA_DIR / project_id


def _assets(project_id: str) -> Path:
    path = _dir(project_id) / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(project: dict[str, Any]) -> None:
    project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = _dir(project["id"]) / "project.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_project(project_id: str) -> dict[str, Any]:
    path = _dir(project_id) / "project.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DramaError("项目不存在或已损坏") from exc


def _summary(project: dict[str, Any]) -> dict[str, Any]:
    return {
        key: project.get(key) for key in (
            "id", "title", "status", "stage", "progress", "error",
            "created_at", "updated_at", "options", "novel_chars",
        )
    } | {
        "episodes": [
            {"index": ep.get("index"), "title": ep.get("title"), "status": ep.get("status"),
             "video": ep.get("video"), "video_url": ep.get("video_url"),
             "shots_done": sum(1 for s in ep.get("shots") or [] if s.get("video")),
             "shots_total": len(ep.get("shots") or [])}
            for ep in project.get("episodes") or []
        ],
    }


def list_projects() -> list[dict[str, Any]]:
    rows = []
    if not DRAMA_DIR.is_dir():
        return rows
    for path in DRAMA_DIR.glob("*/project.json"):
        try:
            project = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append(_summary(project))
    return sorted(rows, key=lambda item: item.get("created_at") or "", reverse=True)


def asset_path(project_id: str, name: str) -> Path:
    safe = Path(name).name
    if safe != name or not re.fullmatch(r"[A-Za-z0-9_.-]+", safe):
        raise DramaError("文件不存在")
    path = _assets(project_id) / safe
    if not path.is_file():
        raise DramaError("文件不存在")
    return path


# ---------------------------------------------------------------- 工具

def _log(project: dict[str, Any], message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    project.setdefault("logs", []).append(line)
    project["logs"] = project["logs"][-500:]
    task_id = project.get("task_id")
    if task_id:
        task = task_manager.get(task_id)
        if task:
            task.log(message)


def _check_cancel(project: dict[str, Any]) -> None:
    if project.get("cancel_requested"):
        raise DramaError("项目已取消")
    task_id = project.get("task_id")
    if task_id:
        task = task_manager.get(task_id)
        if task is not None and task.cancelled:
            raise DramaError("项目已取消")


def _retry(project: dict[str, Any], label: str, fn, attempts: int | None = None):
    """失败自动换 Key 重试直到成功（上限 max_attempts，防全 Key 失效死循环）。"""
    limit = attempts or int(project["options"].get("max_attempts") or 8)
    last: Exception | None = None
    for attempt in range(1, limit + 1):
        _check_cancel(project)
        try:
            return fn()
        except DramaError:
            raise
        except Exception as exc:  # noqa: BLE001 - 上游错误类型多，统一换 Key 重试
            last = exc
            _log(project, f"{label} 第 {attempt}/{limit} 次失败：{str(exc)[:160]}")
            if attempt < limit:
                _log(project, f"{label} 自动换 Key，10 秒后重试…")
                time.sleep(10)
            _save(project)
    raise DramaError(f"{label} 连续 {limit} 次失败：{str(last)[:200]}。可检查 Key 池后点「继续」断点续跑。")


def _llm_json(project: dict[str, Any], label: str, system: str, user: str) -> dict[str, Any]:
    def call() -> dict[str, Any]:
        from app.services.understanding import _request_json

        return _request_json([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    return _retry(project, label, call)


def _data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _to_jpeg(source: Path, target: Path) -> Path:
    """生成图统一转 JPEG：锚定图/首帧都要反复当参考图上传，JPEG 体积只有 PNG 零头。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-q:v", "3", str(target)],
        capture_output=True, timeout=120, check=False,
    )
    if result.returncode != 0 or not target.is_file():
        shutil.copyfile(source, target)
    return target


def _shot_characters(project: dict[str, Any], shot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cid in shot.get("characters") or []:
        for character in project.get("characters") or []:
            if character.get("id") == cid:
                rows.append(character)
    if not rows and project.get("characters"):
        rows = [project["characters"][0]]
    return rows


def _cast_description(cast: list[dict[str, Any]]) -> str:
    return "；".join(
        f"{c['name']}（{c.get('appearance')}，{c.get('clothing')}）" for c in cast
    )


# ---------------------------------------------------------------- 步骤一：小说分析

ANALYZE_SYSTEM = (
    "你是资深短剧编剧与制片人，擅长把小说改编成适合 AI 视频生成的竖屏短剧。"
    "只输出严格 JSON，不要输出任何解释。"
)


def _analyze(project: dict[str, Any]) -> None:
    if project.get("characters") and project.get("episodes"):
        return
    novel = (_dir(project["id"]) / "novel.txt").read_text(encoding="utf-8")
    options = project["options"]
    count = int(options["episodes_count"])
    user = (
        f"把下面这部小说改编为 {count} 集 AI 短剧。\n"
        "要求：\n"
        "1. title：短剧名；logline：一句话卖点。\n"
        "2. characters：主要角色（不超过 4 个），每个角色含 name、age（具体数字）、gender、"
        "appearance（固定外貌特征：脸型、眼型、发型发色、体型、标志物，80 字内，全剧不变）、"
        "clothing（固定服装：款式颜色材质，50 字内）、personality（气质，20 字内）。\n"
        "3. episodes：每集含 index、title、summary（150 字内，开头 3 秒必须有强冲突钩子，"
        "结尾留悬念）、source_excerpt（该集对应的小说原文关键片段，800 字内，供分镜使用）。\n"
        "4. 角色外貌和服装描述要具体、可视化，禁止「年轻」「漂亮」这类模糊词。\n\n"
        f"小说全文：\n{novel[:80000]}"
    )
    result = _llm_json(project, "分析小说", ANALYZE_SYSTEM, user)
    project["title"] = project["title"] or str(result.get("title") or "AI 短剧")[:60]
    project["logline"] = str(result.get("logline") or "")
    characters = []
    for index, item in enumerate(result.get("characters") or []):
        characters.append({
            "id": f"c{index + 1}",
            "name": str(item.get("name") or f"角色{index + 1}")[:30],
            "age": item.get("age"),
            "gender": str(item.get("gender") or ""),
            "appearance": str(item.get("appearance") or ""),
            "clothing": str(item.get("clothing") or ""),
            "personality": str(item.get("personality") or ""),
            "profile_id": "",
            "anchor": "",
            "anchor_url": "",
            "status": "pending",
        })
    if not characters:
        raise DramaError("小说分析没有提取到角色，请补充更完整的小说内容")
    raw_episodes = result.get("episodes") or []
    episodes = []
    for index in range(count):
        item = raw_episodes[index] if index < len(raw_episodes) else {}
        episodes.append({
            "index": index + 1,
            "title": str(item.get("title") or f"第 {index + 1} 集")[:60],
            "summary": str(item.get("summary") or ""),
            "source_excerpt": str(item.get("source_excerpt") or ""),
            "shots": [],
            "video": "",
            "srt": "",
            "status": "pending",
        })
    project["characters"] = characters
    project["episodes"] = episodes
    _save(project)


# ---------------------------------------------------------------- 步骤二：角色定妆（锚定图 + 档案）

def _character_payload(project: dict[str, Any], character: dict[str, Any], anchor: str = "") -> dict[str, Any]:
    return {
        "id": character["profile_id"],
        "name": f"{project['title']}·{character['name']}",
        "fixed_features": character.get("appearance") or "",
        "clothing": character.get("clothing") or "",
        "age_rules": f"年龄固定为 {character.get('age') or '成年'}，全剧不变",
        "references": [{"name": anchor}] if anchor else [],
    }


def _build_characters(project: dict[str, Any]) -> None:
    ratio = project["options"]["aspect_ratio"]
    for character in project.get("characters") or []:
        if character.get("profile_id") and character.get("anchor"):
            continue
        _check_cancel(project)
        name = character["name"]
        if not character.get("profile_id"):
            character["profile_id"] = f"drama-{project['id']}-{character['id']}"
            profiles.save_profile("characters", _character_payload(project, character))
            _save(project)
        profile_id = character["profile_id"]

        def draw() -> dict[str, Any]:
            return ai_media.generate_image(
                prompt=(
                    f"角色定妆照：{name}，{character.get('age') or ''} 岁，{character.get('gender')}。"
                    f"{character.get('appearance')}。{character.get('clothing')}。"
                    f"气质：{character.get('personality')}。"
                    "正面半身像，面部特征清晰无遮挡，光线均匀，纯色简洁背景，"
                    "这是一张用于全剧角色一致性锁定的锚定图。"
                ),
                size="2K",
                ratio=ratio,
                character_profile_id=profile_id,
            )

        result = _retry(project, f"角色定妆·{name}", draw)
        # 转 JPEG 后写入人物档案参考图，后续所有镜头自动携带该锚定图
        jpeg = _to_jpeg(
            ai_media.GENERATED_IMAGE_DIR / result["name"],
            _assets(project["id"]) / f"anchor_{character['id']}.jpg",
        )
        reference = profiles.save_reference(jpeg.read_bytes(), jpeg.name)
        profiles.save_profile(
            "characters", _character_payload(project, character, reference["name"]), profile_id=profile_id,
        )
        character["anchor"] = reference["name"]
        character["anchor_url"] = f"/api/profiles/references/{reference['name']}"
        character["status"] = "done"
        _log(project, f"角色「{name}」锚定图已生成并写入人物档案")
        _save(project)


# ---------------------------------------------------------------- 步骤三：分集分镜

STORYBOARD_SYSTEM = (
    "你是短剧分镜师，为 AI 图生视频设计分镜。每个镜头只有 3-12 秒，"
    "只能表现一个缓慢、连续、简单的动作。只输出严格 JSON。"
)


def _storyboard(project: dict[str, Any]) -> None:
    options = project["options"]
    style = STYLES.get(options.get("style") or "realistic", STYLES["realistic"])
    for episode in project.get("episodes") or []:
        if episode.get("shots"):
            continue
        _check_cancel(project)
        character_brief = [
            {"id": c["id"], "name": c["name"], "appearance": c["appearance"], "clothing": c["clothing"]}
            for c in project["characters"]
        ]
        user = (
            f"为短剧《{project['title']}》第 {episode['index']} 集「{episode['title']}」设计分镜。\n"
            f"本集剧情：{episode['summary']}\n"
            f"原文片段：{episode.get('source_excerpt') or '（无，按剧情创作）'}\n"
            f"角色表：{json.dumps(character_brief, ensure_ascii=False)}\n"
            f"整体画风：{style}\n"
            "要求：\n"
            f"1. 共 {options['shots_per_episode']} 个镜头，输出 shots 数组。\n"
            "2. 每个镜头含：index、scene（场景，含时间光线）、action（一个缓慢连续的简单动作，"
            "避免打斗快跑等剧烈运动）、expression（情绪表情）、camera（固定/缓慢推近/缓慢拉远/缓慢横移 之一）、"
            f"seconds（统一 {options['seconds_per_shot']}）、dialogue（该镜台词或旁白，30 字内，可为空）、"
            "characters（出镜角色 id 数组，尽量单人出镜，最多 2 人）。\n"
            "3. 第 1 镜必须是强冲突钩子画面，镜头之间剧情连贯。\n"
        )
        result = _llm_json(project, f"分镜·第 {episode['index']} 集", STORYBOARD_SYSTEM, user)
        shots = []
        for index, item in enumerate(result.get("shots") or []):
            shots.append({
                "index": index + 1,
                "scene": str(item.get("scene") or ""),
                "action": str(item.get("action") or ""),
                "expression": str(item.get("expression") or ""),
                "camera": str(item.get("camera") or "固定"),
                "seconds": int(options["seconds_per_shot"]),
                "dialogue": str(item.get("dialogue") or ""),
                "characters": [str(cid) for cid in item.get("characters") or []][:2],
                "frame": "",
                "video": "",
                "attempts": 0,
                "status": "pending",
            })
        if not shots:
            raise DramaError(f"第 {episode['index']} 集分镜为空")
        episode["shots"] = shots
        episode["status"] = "storyboard"
        _save(project)


# ---------------------------------------------------------------- 步骤四：逐镜首帧图

def _frame_prompt(project: dict[str, Any], episode: dict[str, Any], shot: dict[str, Any]) -> str:
    style = STYLES.get(project["options"].get("style") or "realistic", STYLES["realistic"])
    cast = _shot_characters(project, shot)
    return (
        f"短剧《{project['title']}》第 {episode['index']} 集第 {shot['index']} 镜首帧。"
        f"画风：{style}。场景：{shot['scene']}。"
        f"出镜角色：{_cast_description(cast) or '按人物档案'}。"
        "严格保持人物档案与参考图的固定面部特征、发型与服装，禁止换脸。"
        f"画面定格在动作开始瞬间：{shot['action']}。表情：{shot['expression']}。"
        "电影感构图，主体清晰，为后续图生视频的首帧。"
    )


def _build_frames(project: dict[str, Any]) -> None:
    ratio = project["options"]["aspect_ratio"]
    for episode in project.get("episodes") or []:
        for shot in episode.get("shots") or []:
            if shot.get("frame"):
                continue
            _check_cancel(project)
            cast = _shot_characters(project, shot)
            main = cast[0] if cast else {}
            extra_refs: list[str] = []
            for character in cast[1:]:
                try:
                    profile = profiles.get_profile("characters", character["profile_id"], include_data=True)
                    extra_refs.extend(
                        item["data_uri"] for item in profile.get("references") or [] if item.get("data_uri")
                    )
                except Exception:  # noqa: BLE001
                    continue
            label = f"首帧·E{episode['index']}S{shot['index']}"

            def draw() -> dict[str, Any]:
                return ai_media.generate_image(
                    prompt=_frame_prompt(project, episode, shot),
                    size="2K",
                    ratio=ratio,
                    images=extra_refs[:2],
                    character_profile_id=main.get("profile_id") or "",
                )

            result = _retry(project, label, draw)
            frame_name = f"e{episode['index']}s{shot['index']}.jpg"
            _to_jpeg(ai_media.GENERATED_IMAGE_DIR / result["name"], _assets(project["id"]) / frame_name)
            shot["frame"] = frame_name
            shot["frame_url"] = f"/api/drama/projects/{project['id']}/files/{frame_name}"
            shot["status"] = "framed"
            shot["attempts"] = int(shot.get("attempts") or 0) + 1
            _save(project)
        episode["status"] = "framed"
        _save(project)


# ---------------------------------------------------------------- 步骤五：逐镜视频（失败换 Key 直到成功）

def _video_prompt(project: dict[str, Any], shot: dict[str, Any]) -> str:
    prompt = (
        f"从首帧开始，{shot['action']}。运镜：{shot['camera']}，缓慢稳定。"
        f"表情变化：{shot['expression']}。"
        "人物外貌、服装、场景光线与首帧完全一致，不变脸、不换人、不切换场景，"
        "动作缓慢自然，无快速剧烈运动。"
    )
    if shot.get("dialogue"):
        prompt += f"角色说台词时口型自然：「{shot['dialogue']}」。"
    return prompt


def _wait_video_task(project: dict[str, Any], task_id: str) -> str:
    """等待现有 AI 视频任务结束，返回生成的视频文件名。"""
    while True:
        _check_cancel(project)
        task = task_manager.get(task_id)
        if task is None:
            raise _RetryableError("视频任务丢失")
        if task.status in {"success", "partial"}:
            url = str((task.result or {}).get("video_url") or "")
            name = Path(url).name
            if not name.startswith("ai_"):
                raise _RetryableError("视频任务未返回文件")
            return name
        if task.status in {"failed", "cancelled"}:
            message = str((task.error or {}).get("message") or "视频生成失败")
            raise _RetryableError(message)
        time.sleep(3)


def _build_videos(project: dict[str, Any]) -> None:
    ratio = project["options"]["aspect_ratio"]
    for episode in project.get("episodes") or []:
        for shot in episode.get("shots") or []:
            if shot.get("video"):
                continue
            if not shot.get("frame"):
                raise DramaError(f"E{episode['index']}S{shot['index']} 缺少首帧，无法生成视频")
            _check_cancel(project)
            cast = _shot_characters(project, shot)
            main = cast[0] if cast else {}
            frame_path = _assets(project["id"]) / shot["frame"]
            if not frame_path.is_file():
                shot["frame"] = ""
                _save(project)
                raise DramaError(f"E{episode['index']}S{shot['index']} 首帧文件丢失，请点「继续」重新生成")
            label = f"视频·E{episode['index']}S{shot['index']}"

            def make() -> str:
                created = ai_media.start_video_generation(
                    prompt=_video_prompt(project, shot),
                    seconds=int(shot["seconds"]),
                    aspect_ratio=ratio,
                    first_frame=_data_uri(frame_path),
                    save_to_library=False,
                    character_profile_id=main.get("profile_id") or "",
                )
                name = _wait_video_task(project, created["task_id"])
                source = ai_media.GENERATED_VIDEO_DIR / name
                target = _assets(project["id"]) / f"e{episode['index']}s{shot['index']}.mp4"
                shutil.copyfile(source, target)
                return target.name

            video_name = _retry(project, label, make)
            shot["video"] = video_name
            shot["video_url"] = f"/api/drama/projects/{project['id']}/files/{video_name}"
            shot["status"] = "done"
            shot["attempts"] = int(shot.get("attempts") or 0) + 1
            _log(project, f"{label} 完成")
            _save(project)
        episode["status"] = "videos"
        _save(project)


# ---------------------------------------------------------------- 步骤六：合成单集 + 台词字幕

def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _compose(project: dict[str, Any]) -> None:
    for episode in project.get("episodes") or []:
        if episode.get("video"):
            continue
        _check_cancel(project)
        shots = episode.get("shots") or []
        lines = []
        srt_blocks = []
        cursor = 0.0
        for index, shot in enumerate(shots):
            clip = _assets(project["id"]) / shot["video"]
            if not clip.is_file():
                raise DramaError(f"第 {episode['index']} 集镜头 {shot['index']} 视频缺失")
            lines.append(f"file '{clip.name}'")
            duration = float(probe_video(clip).get("duration") or shot["seconds"])
            if shot.get("dialogue"):
                srt_blocks.append(
                    f"{index + 1}\n{_srt_time(cursor)} --> {_srt_time(cursor + duration)}\n{shot['dialogue']}\n"
                )
            cursor += duration
        assets = _assets(project["id"])
        concat = assets / f"e{episode['index']}_concat.txt"
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output = assets / f"episode_{episode['index']}.mp4"
        srt = assets / f"episode_{episode['index']}.srt"
        srt.write_text("\n".join(srt_blocks), encoding="utf-8")

        def run(copy: bool) -> None:
            command = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat)]
            if copy:
                command += ["-c", "copy"]
            else:
                command += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an"]
            command.append(str(output))
            result = subprocess.run(command, capture_output=True, timeout=1800, check=False)
            if result.returncode != 0 or not output.is_file():
                raise DramaError(f"ffmpeg 合成失败：{result.stderr.decode('utf-8', 'ignore')[:200]}")

        try:
            run(copy=True)
        except DramaError:
            run(copy=False)
        if project["options"].get("burn_subtitles") and srt_blocks:
            burned = assets / f"episode_{episode['index']}_subtitled.mp4"
            result = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(output),
                 "-vf", f"subtitles={srt.name}", "-c:a", "copy", str(burned)],
                capture_output=True, timeout=1800, cwd=str(assets), check=False,
            )
            if result.returncode == 0 and burned.is_file():
                burned.replace(output)
            else:
                _log(project, f"第 {episode['index']} 集字幕烧录失败，保留无字幕成片与 SRT 文件")
        episode["video"] = output.name
        episode["video_url"] = f"/api/drama/projects/{project['id']}/files/{output.name}"
        episode["srt"] = srt.name
        episode["srt_url"] = f"/api/drama/projects/{project['id']}/files/{srt.name}"
        episode["duration"] = round(cursor, 1)
        episode["status"] = "done"
        _log(project, f"第 {episode['index']} 集合成完成（{cursor:.0f} 秒）")
        _save(project)


# ---------------------------------------------------------------- 流水线主控

def _set_stage(project: dict[str, Any], stage: str, progress: float) -> None:
    project["stage"] = stage
    project["progress"] = round(progress, 1)
    task_id = project.get("task_id")
    if task_id:
        task = task_manager.get(task_id)
        if task:
            task.progress = progress
    _save(project)


def _shot_ratio(project: dict[str, Any], key: str) -> float:
    shots = [s for ep in project["episodes"] for s in ep.get("shots") or []]
    if not shots:
        return 0.0
    return sum(1 for s in shots if s.get(key)) / len(shots)


def _run(project_id: str) -> None:
    project = load_project(project_id)
    task_id = project.get("task_id")
    task = task_manager.get(task_id) if task_id else None
    try:
        project["status"] = "running"
        _set_stage(project, "分析小说：提取角色与分集大纲", 2)
        _analyze(project)
        _set_stage(project, "角色定妆：生成锚定图与人物档案", 8)
        _build_characters(project)
        _set_stage(project, "分集分镜：生成逐镜剧本", 18)
        _storyboard(project)
        while _shot_ratio(project, "frame") < 1:
            _set_stage(project, "逐镜生成首帧图", 20 + 35 * _shot_ratio(project, "frame"))
            _build_frames(project)
        while _shot_ratio(project, "video") < 1:
            _set_stage(project, "逐镜生成视频（失败自动换 Key）", 55 + 35 * _shot_ratio(project, "video"))
            _build_videos(project)
        _set_stage(project, "合成单集与台词字幕", 92)
        _compose(project)
        project["status"] = "success"
        _set_stage(project, "全部完成", 100)
        _log(project, "短剧全部集数生成完成")
        if task:
            task.result = {"project_id": project_id}
            task.finish("success")
    except Exception as exc:  # noqa: BLE001
        cancelled = "已取消" in str(exc)
        project["status"] = "cancelled" if cancelled else "failed"
        project["error"] = "" if cancelled else str(exc)[:400]
        _log(project, ("项目已取消" if cancelled else f"项目中断：{str(exc)[:200]}") + "，可点「继续」断点续跑")
        _save(project)
        if task:
            if not cancelled:
                task.error = {"code": "drama_failed", "message": str(exc)[:300],
                              "action": "在 AI 短剧页面点「继续」可从断点续跑。"}
            task.finish("cancelled" if cancelled else "failed")
    finally:
        with _LOCK:
            THREADS.pop(project_id, None)


def _spawn(project: dict[str, Any]) -> None:
    with _LOCK:
        thread = THREADS.get(project["id"])
        if thread and thread.is_alive():
            return
        task = task_manager.create("ai", f"AI 短剧 · {project['title'] or project['id']}")
        project["task_id"] = task.id
        _save(project)
        thread = threading.Thread(target=_run, args=(project["id"],), daemon=True)
        THREADS[project["id"]] = thread
        thread.start()


def create_project(novel: str, title: str, options: dict[str, Any]) -> dict[str, Any]:
    if not AI_POOL.has_keys():
        raise DramaError("没有配置 AI API Key")
    text = (novel or "").strip()
    if len(text) < 200:
        raise DramaError("小说内容太短（至少 200 字），请提供完整小说内容")
    if len(text) > 300000:
        raise DramaError("小说内容过长（上限 30 万字）")
    project_id = uuid.uuid4().hex[:12]
    project = {
        "id": project_id,
        "title": (title or "").strip()[:60],
        "logline": "",
        "status": "pending",
        "stage": "排队中",
        "progress": 0.0,
        "error": "",
        "novel_chars": len(text),
        "options": options,
        "characters": [],
        "episodes": [],
        "logs": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_id": "",
        "cancel_requested": False,
    }
    folder = _dir(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "novel.txt").write_text(text, encoding="utf-8")
    _assets(project_id)
    _save(project)
    _spawn(project)
    return _summary(project)


def resume_project(project_id: str) -> dict[str, Any]:
    project = load_project(project_id)
    with _LOCK:
        thread = THREADS.get(project_id)
        if thread and thread.is_alive():
            raise DramaError("项目正在运行中")
    if project.get("status") == "success":
        raise DramaError("项目已完成")
    project["cancel_requested"] = False
    project["error"] = ""
    _spawn(project)
    return _summary(project)


def cancel_project(project_id: str) -> dict[str, Any]:
    project = load_project(project_id)
    project["cancel_requested"] = True
    task_id = project.get("task_id")
    if task_id:
        task_manager.cancel(task_id)
    _save(project)
    return _summary(project)


def delete_project(project_id: str) -> None:
    with _LOCK:
        thread = THREADS.get(project_id)
        if thread and thread.is_alive():
            raise DramaError("项目正在运行，请先取消")
    folder = _dir(project_id)
    if folder.is_dir():
        shutil.rmtree(folder)


def project_detail(project_id: str) -> dict[str, Any]:
    project = load_project(project_id)
    project.pop("cancel_requested", None)
    return project
