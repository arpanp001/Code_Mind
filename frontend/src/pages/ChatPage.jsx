// frontend/src/pages/ChatPage.jsx
// COMPLETE REPLACEMENT

import { useEffect, useRef, useState }    from 'react'
import { useParams, useNavigate }          from 'react-router-dom'
import { Trash2, ChevronDown } from 'lucide-react'
import Sidebar       from '@/components/layout/Sidebar'
import MessageBubble from '@/components/chat/MessageBubble'
import SourceCard    from '@/components/chat/SourceCard'
import QueryInput    from '@/components/chat/QueryInput'
import ProjectSummary from '@/components/chat/ProjectSummary'
import { useChatContext } from '@/context/ChatContext'
import { ingestService }  from '@/services/ingestService'
import { timeAgo } from '@/utils/formatters'
import {
  exportChatAsMarkdown,
  exportChatAsHTML,
  downloadMarkdown
} from '@/utils/exportChat'
import { Download } from 'lucide-react'

export default function ChatPage() {
  const { projectId }   = useParams()
  const navigate         = useNavigate()
  const [project, setProject]   = useState(null)
  const [loadingP, setLoadingP] = useState(false)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const endRef = useRef(null)
  const scrollAreaRef = useRef(null)
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  

  // Chat state persists globally — survives tab switching
  const { messages, isLoading, sendMessage, clearChat } =
    useChatContext(projectId)

  // Load project info when projectId changes
  useEffect(() => {
    if (projectId) loadProject()
    else setProject(null)
  }, [projectId])

  // Auto-scroll to latest message
  useEffect(() => {
    if (!endRef.current) return
    // Use requestAnimationFrame to ensure DOM has updated
    const frame = requestAnimationFrame(() => {
      endRef.current?.scrollIntoView({
        behavior: 'smooth',
        block:    'end',
      })
    })
    return () => cancelAnimationFrame(frame)
  }, [messages.length, isLoading])

  const handleScroll = () => {
  const el = scrollAreaRef.current

  if (!el) return

  const distance =
    el.scrollHeight - el.scrollTop - el.clientHeight

  setShowScrollBtn(distance > 200)
}


const scrollToBottom = () => {
  endRef.current?.scrollIntoView({
    behavior: 'smooth',
  })

  setShowScrollBtn(false)
}

  const loadProject = async () => {
    try {
      setLoadingP(true)
      const data = await ingestService.getStatus(projectId)
      setProject(data)
    } catch {
      navigate('/')
    } finally {
      setLoadingP(false)
    }
  }

  // Sources from the most recent assistant message
  const latestSources = [...messages]
    .reverse()
    .find(m => m.role === 'assistant' && m.sources?.length > 0)
    ?.sources || []

  const groupedSources = latestSources.reduce((acc, source) => {
    const existing = acc.find(s => s.file_path === source.file_path)
    if (existing) {
      // Keep the highest relevance score, merge line ranges
      if (source.relevance_score > existing.relevance_score) {
        existing.relevance_score = source.relevance_score
        existing.code            = source.code
        existing.start_line      = source.start_line
        existing.end_line        = source.end_line
      }
      // Track all line ranges for this file
      existing.all_ranges = existing.all_ranges || []
      existing.all_ranges.push({
        start: source.start_line,
        end:   source.end_line,
        code:  source.code,
        score: source.relevance_score,
      })
    } else {
      acc.push({
        ...source,
        all_ranges: [{
          start: source.start_line,
          end:   source.end_line,
          code:  source.code,
          score: source.relevance_score,
        }]
      })
    }
    return acc
  }, [])

  // ── No project selected ────────────────────────────────────────────────────
  if (!projectId) {
    return (
      <div className="flex h-screen pt-14">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center
                        text-center space-y-4 px-6">
          <div className="text-5xl">💬</div>
          <h2 className="text-white font-semibold text-lg">
            Select a project to start chatting
          </h2>
          <p className="text-gray-500 text-sm max-w-xs">
            Choose a project from the sidebar, or upload a new codebase.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen pt-14">
      <Sidebar />

      <div className="relative flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Project header */}
       <div className="border-b border-gray-800 px-5 py-3 shrink-0
                        bg-gray-950/50">
          <div className="flex items-center justify-between">
            {loadingP ? (
              <div className="h-4 w-48 bg-gray-800 rounded animate-pulse" />
            ) : project ? (
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <span className="text-xl shrink-0">
                  {project.source_type === 'github' ? '⚡' : '📦'}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h1 className="text-sm font-semibold text-white truncate">
                      {project.name}
                    </h1>
                    <span className={`text-xs px-2 py-0.5 rounded-full
                                      font-medium shrink-0
                                      ${project.status === 'ready'
                                        ? 'bg-green-400/10 text-green-400'
                                        : 'bg-yellow-400/10 text-yellow-400'}`}>
                      {project.status}
                    </span>
                  </div>
                  {/* Metadata row */}
                  <div className="flex items-center gap-3 mt-0.5 flex-wrap">
                    <span className="text-xs text-gray-500">
                      📁 {project.file_count} files
                    </span>
                    <span className="text-xs text-gray-500">
                      🧩 {project.chunk_count} chunks
                    </span>
                    {messages.length > 0 && (
                      <span className="text-xs text-indigo-400">
                        💬 {Math.floor(messages.filter(m =>
                          m.role !== 'error').length / 2)} exchanges
                      </span>
                    )}
                    {project.source_type === 'github' && project.source_url && (
                      <a
                        href={project.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-gray-600 hover:text-indigo-400
                                  transition-colors truncate max-w-48"
                      >
                        🔗 {project.source_url.replace('https://github.com/', '')}
                      </a>
                    )}
                    <span className="text-xs text-gray-600">
                      🕐 {timeAgo(project.created_at)}
                    </span>
                  </div>
                </div>
              </div>
            ) : null}

            <div className="flex items-center gap-2 shrink-0">
              {/* Re-index button */}
              {project?.status === 'ready' && project?.source_type === 'github' && (
                <button
                  onClick={async () => {
                    if (!confirm('Re-index this project? This will clear existing embeddings and rebuild from scratch.')) return
                    try {
                      await ingestService.reindexProject(projectId)
                      // Navigate home to show progress
                      navigate('/')
                      // Start polling for this project
                      setTimeout(() => navigate(`/chat/${projectId}`), 500)
                    } catch (e) {
                      alert('Re-index failed: ' + (e.userMessage || e.message))
                    }
                  }}
                  className="btn-ghost text-xs"
                  title="Re-index project embeddings"
                >
                  🔄
                </button>
              )}

               {messages.length > 0 && (
  <div className="relative">

    <button
      onClick={() => setShowExportMenu(!showExportMenu)}
      className="btn-ghost text-xs flex items-center gap-1"
    >
      <Download className="w-3.5 h-3.5" />
      <span className="hidden sm:inline">Export</span>
    </button>

    {showExportMenu && (
      <div
        className="absolute right-0 top-full mt-1 bg-gray-800
                   border border-gray-700 rounded-xl shadow-xl
                   overflow-hidden z-50 min-w-36"
      >

        {/* Markdown */}
        <button
          onClick={() => {
            const md = exportChatAsMarkdown(
              messages,
              project?.name || 'chat'
            )

            downloadMarkdown(
              md,
              `${project?.name || 'chat'}-chat.md`
            )

            setShowExportMenu(false)
          }}
          className="w-full text-left px-4 py-2.5 text-sm
                     text-gray-300 hover:bg-gray-700"
        >
          📄 Markdown
        </button>

        {/* HTML */}
        <button
          onClick={() => {
            const html = exportChatAsHTML(
              messages,
              project?.name || 'chat'
            )

            const blob = new Blob([html], {
              type: 'text/html'
            })

            const url = URL.createObjectURL(blob)

            const a = document.createElement('a')
            a.href = url
            a.download = `${project?.name || 'chat'}-chat.html`
            a.click()

            URL.revokeObjectURL(url)

            setShowExportMenu(false)
          }}
          className="w-full text-left px-4 py-2.5 text-sm
                     text-gray-300 hover:bg-gray-700"
        >
          🌐 HTML
        </button>

      </div>
    )}

  </div>
)}
              <button
                onClick={() => clearChat()}
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={scrollAreaRef}
          onScroll={handleScroll}
          className="flex-1 overflow-y-auto px-5 py-4 space-y-3"
        >
    {messages.length === 0 ? (
      <div className="
        flex flex-col items-center justify-center
        h-full text-center space-y-4
        px-6 max-w-2xl mx-auto
      ">
        <div className="text-4xl">💬</div>

        <h2 className="text-white font-semibold text-lg">
          {project ? `Chat with ${project.name}` : 'Start chatting'}
        </h2>

        {/* Auto-generated project overview */}
        <ProjectSummary
            projectId={projectId}
            project={project}
        />

        <p className="text-gray-600 text-xs">
          Ask where features are implemented,
          get function explanations,
          or explore the architecture.
        </p>
      </div>
    ) : (
      messages.map(m => (
        <MessageBubble
          key={m.id}
          message={m}
        />
      ))
    )}

          {/* Typing indicator */}
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-7 h-7 rounded-full bg-gray-800 border border-gray-700
                              flex items-center justify-center shrink-0 mt-0.5">
                <span className="text-xs">🧠</span>
              </div>
              <div className="bg-gray-800/80 rounded-2xl rounded-tl-sm
                              px-4 py-3">
                <div className="flex gap-1 items-center">
                  {[0, 1, 2].map(i => (
                    <span
                      key={i}
                      className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {showScrollBtn && (
          <div className="absolute bottom-40 right-6 z-10">
            <button
              onClick={scrollToBottom}
              className="
              bg-indigo-600 hover:bg-indigo-700
              text-white rounded-full
                w-8 h-8 flex items-center justify-center
                shadow-lg transition-colors
              "
              title="Scroll to latest"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>
        )}

        
        {groupedSources.length > 0 && (
          <div className="border-t border-gray-800 bg-gray-900/50
                          px-5 py-3 shrink-0 max-h-56 overflow-y-auto">
            <p className="text-xs text-gray-500 font-medium uppercase
                          tracking-wider mb-2">
              Source files ({groupedSources.length}
              {latestSources.length > groupedSources.length &&
                ` · ${latestSources.length} chunks`
              })
            </p>
            <div className="space-y-2">
              {groupedSources.map((s, i) => (
                <SourceCard key={s.file_path} source={s} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* Input */}
        <div className="border-t border-gray-800 p-4 shrink-0 bg-gray-950">
          <QueryInput
            onSend={sendMessage}
            isLoading={isLoading}
            disabled={!projectId || project?.status !== 'ready'}
          />
        </div>
      </div>
    </div>
  )
}