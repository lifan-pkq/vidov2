import { useCallback, useEffect, useRef, useState } from 'react'
import { DownloadSimple, FilmStrip, Play, Plus, Trash, ArrowClockwise, Stop } from '@phosphor-icons/react'
import {
  cancelDramaProject,
  createDramaProject,
  deleteDramaProject,
  errorMessage,
  fetchDramaProject,
  fetchDramaProjects,
  formatDuration,
  resumeDramaProject,
  type DramaProject,
  type DramaProjectSummary,
} from '../api'
import { Alert, Button, Input, PageHeader, Panel, Progress, Select, Switch, TextArea, toast } from '../ui'

const EPISODE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10].map((n) => ({ value: String(n), label: `${n} 集` }))
const SHOT_COUNTS = [3, 4, 5, 6, 8, 10, 12].map((n) => ({ value: String(n), label: `${n} 个镜头 / 集` }))
const SHOT_SECONDS = [4, 5, 6, 8, 10, 12].map((n) => ({ value: String(n), label: `${n} 秒 / 镜头` }))
const RATIOS = [
  { value: '9:16', label: '9:16 竖屏（抖音）' },
  { value: '16:9', label: '16:9 横屏' },
  { value: '3:4', label: '3:4 竖版' },
  { value: '1:1', label: '1:1 方形' },
]
const STYLE_OPTIONS = [
  { value: 'realistic', label: '写实 · 电影感' },
  { value: 'semi', label: '半写实 · 插画' },
  { value: 'anime', label: '日系动漫' },
  { value: 'guofeng', label: '国风工笔' },
]
const ATTEMPT_OPTIONS = [5, 8, 12, 20].map((n) => ({ value: String(n), label: `${n} 次` }))

const STATUS_LABEL: Record<string, string> = {
  pending: '排队中',
  running: '生成中',
  success: '已完成',
  failed: '已中断',
  cancelled: '已取消',
}

function StatusTag({ status }: { status: string }) {
  const color = status === 'success'
    ? 'text-success bg-success/10'
    : status === 'failed'
      ? 'text-danger bg-danger/10'
      : status === 'running'
        ? 'text-accent bg-accent/10'
        : 'text-ink-3 bg-fill-deep'
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs ${color}`}>{STATUS_LABEL[status] || status}</span>
}

export default function AIDramaPage() {
  const [title, setTitle] = useState('')
  const [novel, setNovel] = useState('')
  const [episodesCount, setEpisodesCount] = useState('2')
  const [shotsPerEpisode, setShotsPerEpisode] = useState('6')
  const [secondsPerShot, setSecondsPerShot] = useState('5')
  const [aspectRatio, setAspectRatio] = useState('9:16')
  const [style, setStyle] = useState('realistic')
  const [burnSubtitles, setBurnSubtitles] = useState(false)
  const [maxAttempts, setMaxAttempts] = useState('8')
  const [creating, setCreating] = useState(false)
  const [projects, setProjects] = useState<DramaProjectSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [detail, setDetail] = useState<DramaProject | null>(null)
  const [loadError, setLoadError] = useState('')
  const timerRef = useRef(0)

  const loadProjects = useCallback(async () => {
    try {
      setProjects(await fetchDramaProjects())
      setLoadError('')
    } catch (error) {
      setLoadError(errorMessage(error))
    }
  }, [])

  const loadDetail = useCallback(async (id: string) => {
    try {
      setDetail(await fetchDramaProject(id))
    } catch (error) {
      toast.error(errorMessage(error))
    }
  }, [])

  useEffect(() => {
    loadProjects()
    timerRef.current = window.setInterval(() => {
      loadProjects()
    }, 5000)
    return () => window.clearInterval(timerRef.current)
  }, [loadProjects])

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      return
    }
    loadDetail(selectedId)
    const timer = window.setInterval(() => {
      loadDetail(selectedId)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [selectedId, loadDetail])

  const submit = async () => {
    if (novel.trim().length < 200) {
      toast.error('请粘贴完整小说内容（至少 200 字）')
      return
    }
    setCreating(true)
    try {
      const project = await createDramaProject({
        novel,
        title,
        episodes_count: Number(episodesCount),
        shots_per_episode: Number(shotsPerEpisode),
        seconds_per_shot: Number(secondsPerShot),
        aspect_ratio: aspectRatio,
        style,
        burn_subtitles: burnSubtitles,
        max_attempts: Number(maxAttempts),
      })
      toast.success('短剧项目已创建，开始自动生成')
      setSelectedId(project.id)
      setNovel('')
      setTitle('')
      loadProjects()
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setCreating(false)
    }
  }

  const action = async (kind: 'resume' | 'cancel' | 'delete', id: string) => {
    try {
      if (kind === 'resume') {
        await resumeDramaProject(id)
        toast.success('已从断点继续生成')
      } else if (kind === 'cancel') {
        await cancelDramaProject(id)
        toast.success('已请求取消，产物保留可续跑')
      } else {
        if (!window.confirm('确定删除该项目？项目目录与成片都会被删除。')) return
        await deleteDramaProject(id)
        if (selectedId === id) setSelectedId('')
        toast.success('项目已删除')
      }
      loadProjects()
      if (selectedId === id) loadDetail(id)
    } catch (error) {
      toast.error(errorMessage(error))
    }
  }

  const running = detail?.status === 'running' || detail?.status === 'pending'

  return (
    <div>
      <PageHeader
        title="AI 短剧"
        desc="粘贴完整小说 → 自动分集分镜 → 角色定妆锚定图 → 逐镜首帧 → 图生视频 → 合成单集。生成失败自动换 Key 重试，断点可续跑。"
      />

      <Panel title="新建短剧项目" className="mb-6">
        <div className="flex flex-col gap-4">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="text-xs text-ink-3">剧名（可选，留空由 AI 命名）
              <Input value={title} onChange={setTitle} placeholder="例如：重生之我在京城当首富" className="mt-1" />
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-ink-3">集数
                <Select value={episodesCount} onChange={setEpisodesCount} options={EPISODE_COUNTS} className="mt-1" width="100%" />
              </label>
              <label className="text-xs text-ink-3">每集镜头
                <Select value={shotsPerEpisode} onChange={setShotsPerEpisode} options={SHOT_COUNTS} className="mt-1" width="100%" />
              </label>
              <label className="text-xs text-ink-3">镜头时长
                <Select value={secondsPerShot} onChange={setSecondsPerShot} options={SHOT_SECONDS} className="mt-1" width="100%" />
              </label>
              <label className="text-xs text-ink-3">画幅
                <Select value={aspectRatio} onChange={setAspectRatio} options={RATIOS} className="mt-1" width="100%" />
              </label>
            </div>
          </div>
          <label className="text-xs text-ink-3">完整小说内容（至少 200 字，当前 {novel.trim().length} 字）
            <TextArea
              value={novel}
              onChange={setNovel}
              rows={10}
              className="mt-1"
              placeholder="把完整小说粘贴到这里。AI 会自动提取主要角色并生成固定外貌/服装档案，再拆成每集分镜。"
            />
          </label>
          <div className="flex flex-wrap items-end gap-4">
            <label className="text-xs text-ink-3">画风
              <Select value={style} onChange={setStyle} options={STYLE_OPTIONS} className="mt-1" width={160} />
            </label>
            <label className="text-xs text-ink-3" title="单个生成步骤失败后的最大换 Key 重试次数，超限可断点续跑">
              单步重试上限
              <Select value={maxAttempts} onChange={setMaxAttempts} options={ATTEMPT_OPTIONS} className="mt-1" width={110} />
            </label>
            <label className="flex items-center gap-2 text-xs text-ink-3">
              <Switch checked={burnSubtitles} onChange={setBurnSubtitles} label="烧录台词字幕" />
              烧录台词字幕
            </label>
            <Button variant="primary" icon={<Plus size={16} />} loading={creating} onClick={submit} className="ml-auto">
              开始生成短剧
            </Button>
          </div>
          <p className="text-xs text-ink-4">
            预计耗时：每镜头需首帧图 + 视频两步生成，请保持页面或稍后回来查看；所有产物实时落盘，中断后点「继续」即可断点续跑。
          </p>
        </div>
      </Panel>

      {loadError && <Alert type="error" className="mb-4">{loadError}</Alert>}

      <div className="mb-6 flex flex-col gap-3">
        <h2 className="section-title">短剧项目（{projects.length}）</h2>
        {projects.length === 0 && <p className="text-sm text-ink-3">还没有项目，先在上方粘贴小说创建第一部短剧。</p>}
        {projects.map((project) => (
          <Panel key={project.id} pad="sm" className={selectedId === project.id ? 'border-accent' : ''}>
            <div className="flex flex-wrap items-center gap-3">
              <FilmStrip size={22} className="text-accent" />
              <button type="button" className="min-w-0 flex-1 text-left" onClick={() => setSelectedId(project.id === selectedId ? '' : project.id)}>
                <div className="flex items-center gap-2">
                  <strong className="truncate text-sm">{project.title || '未命名短剧'}</strong>
                  <StatusTag status={project.status} />
                </div>
                <p className="mt-0.5 truncate text-xs text-ink-3">
                  {project.options.episodes_count} 集 · {project.options.shots_per_episode} 镜/集 · {project.created_at} · {project.stage}
                </p>
              </button>
              <div className="w-32">
                <Progress percent={(project.progress || 0) / 100} status={project.status === 'failed' ? 'error' : project.status === 'success' ? 'success' : 'normal'} />
              </div>
              <div className="flex items-center gap-1.5">
                {(project.status === 'failed' || project.status === 'cancelled') && (
                  <Button size="sm" icon={<ArrowClockwise size={14} />} onClick={() => action('resume', project.id)}>继续</Button>
                )}
                {(project.status === 'running' || project.status === 'pending') && (
                  <Button size="sm" variant="ghost" icon={<Stop size={14} />} onClick={() => action('cancel', project.id)}>取消</Button>
                )}
                <Button size="sm" variant="text" icon={<Trash size={14} />} onClick={() => action('delete', project.id)} />
              </div>
            </div>
            {project.error && <p className="mt-2 text-xs text-danger">{project.error}</p>}
          </Panel>
        ))}
      </div>

      {detail && (
        <div className="flex flex-col gap-5">
          <Panel title={`项目详情 · ${detail.title || detail.id}`} extra={<StatusTag status={detail.status} />}>
            <p className="mb-3 text-sm text-ink-3">{detail.logline || detail.stage}{running && ` · ${detail.stage}`}</p>
            {detail.characters.length > 0 && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {detail.characters.map((character) => (
                  <div key={character.id} className="overflow-hidden rounded-[10px] border border-line">
                    {character.anchor_url
                      ? <img src={character.anchor_url} alt={character.name} className="aspect-[3/4] w-full object-cover" />
                      : <div className="flex aspect-[3/4] w-full items-center justify-center bg-fill text-xs text-ink-4">定妆照生成中…</div>}
                    <div className="p-2.5">
                      <strong className="text-sm">{character.name}</strong>
                      <span className="ml-1.5 text-xs text-ink-3">{character.age}{character.gender}</span>
                      <p className="mt-1 line-clamp-3 text-xs text-ink-3">{character.appearance}</p>
                      <p className="mt-0.5 line-clamp-2 text-xs text-ink-4">{character.clothing}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Panel>

          {detail.episodes.map((episode) => (
            <Panel
              key={episode.index}
              title={`第 ${episode.index} 集 · ${episode.title}`}
              extra={<span className="text-xs text-ink-3">{episode.shots.filter((s) => s.video).length}/{episode.shots.length} 镜完成{episode.duration ? ` · ${formatDuration(episode.duration)}` : ''}</span>}
            >
              {episode.summary && <p className="mb-3 text-xs text-ink-3">{episode.summary}</p>}
              {episode.video_url && (
                <div className="mb-4 flex flex-wrap items-center gap-3">
                  <video src={episode.video_url} controls className="max-h-80 rounded-[10px] border border-line" />
                  <div className="flex flex-col gap-2">
                    <a href={episode.video_url} download className="inline-flex items-center gap-1.5 text-sm text-accent"><DownloadSimple size={15} />下载第 {episode.index} 集成片</a>
                    {episode.srt_url && <a href={episode.srt_url} download className="inline-flex items-center gap-1.5 text-sm text-accent"><DownloadSimple size={15} />下载台词字幕 SRT</a>}
                  </div>
                </div>
              )}
              {episode.shots.length > 0 && (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {episode.shots.map((shot) => (
                    <div key={shot.index} className="overflow-hidden rounded-[10px] border border-line">
                      {shot.video_url
                        ? <video src={shot.video_url} controls preload="metadata" className="aspect-video w-full bg-black object-contain" />
                        : shot.frame_url
                          ? <img src={shot.frame_url} alt={`镜头 ${shot.index}`} className="aspect-video w-full object-cover" />
                          : <div className="flex aspect-video w-full items-center justify-center bg-fill text-xs text-ink-4">待生成</div>}
                      <div className="p-2.5 text-xs">
                        <div className="mb-1 flex items-center justify-between">
                          <strong>镜头 {shot.index} · {shot.seconds}s · {shot.camera}</strong>
                          {shot.video
                            ? <span className="text-success">完成</span>
                            : <span className="text-ink-4">{shot.frame ? '首帧就绪' : '排队中'}</span>}
                        </div>
                        <p className="line-clamp-2 text-ink-3">{shot.scene}｜{shot.action}</p>
                        {shot.dialogue && <p className="mt-1 line-clamp-2 text-ink-2">「{shot.dialogue}」</p>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          ))}

          <Panel title="生成日志" extra={<Button size="sm" variant="text" icon={<Play size={13} />} onClick={() => loadDetail(detail.id)}>刷新</Button>}>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-ink-3">
              {(detail.logs || []).slice().reverse().join('\n') || '暂无日志'}
            </pre>
          </Panel>
        </div>
      )}
    </div>
  )
}
