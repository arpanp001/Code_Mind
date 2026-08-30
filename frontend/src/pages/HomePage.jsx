// frontend/src/pages/HomePage.jsx
//
// Fixes vs previous version:
//   1. Auto-navigates to /chat/{id} when status === 'ready'
//   2. Auto-navigates to /chat/{id} when status === 'exists' AND user
//      clicks "Open Existing" — does NOT auto-navigate immediately on
//      'exists' so the user sees the choice dialog first
//   3. Re-index uses forceReindex() from the hook (not a separate call)
//      which correctly reuses lastInput.current
//   4. useEffect watches status to trigger navigation — replaces the
//      broken pattern of checking pid return value from async functions
//   5. ProcessingSteps uses real SSE via usePipelineProgress hook

import { useState, useEffect } from 'react'
import { useNavigate }         from 'react-router-dom'
import {
  Brain, Zap, Database, MessageSquare,
  Loader2, CheckCircle, X,
} from 'lucide-react'
import GithubInput  from '@/components/upload/GithubInput'
import ZipUpload    from '@/components/upload/ZipUpload'
import { useIngest } from '@/hooks/useIngest'
import { usePipelineProgress } from '@/hooks/usePipelineProgress'

// ── Feature cards ─────────────────────────────────────────────────────────────

const FEATURES = [
  { Icon: Zap,           title: 'Smart Chunking',  desc: 'Function-aware code splitting'      },
  { Icon: Database,      title: 'Vector Search',   desc: 'ChromaDB semantic retrieval'        },
  { Icon: MessageSquare, title: 'Gemini AI',        desc: 'Real answers with file citations'   },
  { Icon: Brain,         title: 'Memory System',   desc: 'Store architecture decisions'       },
]

// ── Processing steps component (uses real SSE) ────────────────────────────────

const STEP_LABELS = {
  cloning:     { label: 'Cloning repository',     icon: '📥' },
  extracting:  { label: 'Extracting ZIP archive', icon: '📦' },
  parsing:     { label: 'Parsing source files',   icon: '🔍' },
  chunking:    { label: 'Chunking code',          icon: '✂️'  },
  embedding:   { label: 'Generating embeddings',  icon: '🧮' },
  storing:     { label: 'Storing in ChromaDB',    icon: '💾' },
  summarizing: { label: 'Generating summary',     icon: '📋' },
  done:        { label: 'Indexing complete!',      icon: '✅' },
  error:       { label: 'Error occurred',         icon: '❌' },
}

const ORDERED_STEPS = [
  'cloning', 'extracting', 'parsing',
  'chunking', 'embedding', 'storing', 'summarizing',
]

function ProcessingSteps({ projectId, isActive }) {
  const { progress } = usePipelineProgress(projectId, isActive)

  const currentIdx = progress
    ? ORDERED_STEPS.indexOf(progress.step)
    : 0

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      {progress?.percent > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>{progress.detail || STEP_LABELS[progress.step]?.label || progress.step}</span>
            <span>{progress.percent}%</span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
        </div>
      )}

      {/* Step list */}
      <div className="space-y-1.5">
        {ORDERED_STEPS.map((step, i) => {
          const info   = STEP_LABELS[step]
          const isDone = i < currentIdx
          const isCurr = i === currentIdx
          return (
            <div
              key={step}
              className={`flex items-center gap-2 text-xs transition-colors
                           ${isDone ? 'text-green-400' :
                             isCurr ? 'text-white'     :
                                      'text-gray-700'  }`}
            >
              <span className="w-5 text-center shrink-0">
                {isDone ? '✓' : info.icon}
              </span>
              <span className="flex-1">{info.label}</span>
              {isCurr && (
                <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full
                                 animate-bounce shrink-0" />
              )}
            </div>
          )
        })}
      </div>

      {/* Error detail */}
      {progress?.step === 'error' && progress.detail && (
        <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-3">
          <p className="text-red-400 text-xs">{progress.detail}</p>
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [tab, setTab] = useState('github')
  const navigate      = useNavigate()

  const {
    projectId,
    status,
    progress,
    error,
    existingProject,
    isLoading,
    ingestGithub,
    ingestZip,
    forceReindex,
    reset,
  } = useIngest()

  // ── Navigate when indexing completes ──────────────────────────────────────
  // Watches status in an effect so navigation is reliable regardless of
  // async timing — fixes the "stuck in processing" issue
  useEffect(() => {
    if (status === 'ready' && projectId) {
      // Small delay so the user sees the "complete" state briefly
      const timer = setTimeout(() => {
        navigate(`/chat/${projectId}`)
      }, 800)
      return () => clearTimeout(timer)
    }
  }, [status, projectId, navigate])

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleGithub = (url, branch) => {
    ingestGithub(url, branch)
    // Navigation is handled by the useEffect above
  }

  const handleZip = (file) => {
    ingestZip(file)
    // Navigation is handled by the useEffect above
  }

  const handleOpenExisting = () => {
    if (existingProject?.project_id) {
      navigate(`/chat/${existingProject.project_id}`)
    }
  }

  const handleReindex = async () => {
    await forceReindex()
    // useEffect will navigate when status becomes 'ready'
  }

  const handleTabChange = (newTab) => {
    setTab(newTab)
    reset()
  }

  // ────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen flex flex-col items-center justify-center
                    px-4 py-20">

      {/* Hero */}
      <div className="text-center mb-10 space-y-3">
        <div className="inline-flex items-center gap-2 bg-indigo-600/10
                        border border-indigo-500/20 text-indigo-400 text-xs
                        font-medium px-3 py-1 rounded-full mb-2">
          <Brain className="w-3.5 h-3.5" />
          RAG-Powered Codebase Intelligence
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-white">
          Code<span className="text-indigo-400">Mind</span>
        </h1>
        <p className="text-gray-400 max-w-md mx-auto text-sm">
          Upload any codebase and chat with it using AI.
          Find functions, understand architecture, store decisions.
        </p>
      </div>

      {/* Upload card */}
      <div className="w-full max-w-lg space-y-4">

        {/* Tab switcher */}
        <div className="flex bg-gray-900 rounded-xl p-1 gap-1">
          {[
            { id: 'github', label: '⚡ GitHub URL'  },
            { id: 'zip',    label: '📦 ZIP Upload'  },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => handleTabChange(t.id)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium
                          transition-colors
                          ${tab === t.id
                            ? 'bg-indigo-600 text-white'
                            : 'text-gray-500 hover:text-gray-300'}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Upload components — hidden while processing to avoid re-submission */}
        {!isLoading && status !== 'exists' && (
          tab === 'github'
            ? <GithubInput onSubmit={handleGithub} isLoading={isLoading} />
            : <ZipUpload   onSubmit={handleZip}    isLoading={isLoading}
                           progress={progress} />
        )}

        {/* ── Status cards ─────────────────────────────────────────────── */}

        {/* Processing */}
        {(status === 'processing' || status === 'uploading') && projectId && (
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-400 shrink-0" />
                <p className="text-white text-sm font-medium">
                  {status === 'uploading'
                    ? `Uploading… ${progress > 0 ? `${progress}%` : ''}`
                    : 'Processing codebase…'
                  }
                </p>
              </div>
              {/* Cancel button */}
              <button
                onClick={async () => {
                  if (!confirm('Cancel indexing? Progress will be lost.')) return
                  try {
                    const { ingestService: svc } = await import('@/services/ingestService')
                    await svc.cancelIndexing?.(projectId)
                  } catch (_) {}
                  reset()
                }}
                className="text-xs text-gray-600 hover:text-red-400 transition-colors
                           border border-gray-700 hover:border-red-400 px-2 py-1 rounded"
              >
                ✕ Cancel
              </button>
            </div>
            <ProcessingSteps
              projectId={projectId}
              isActive={status === 'processing'}
            />
          </div>
        )}

        {/* Ready */}
        {status === 'ready' && (
          <div className="card border-green-500/30 bg-green-500/5
                          flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />
            <p className="text-green-400 text-sm font-medium">
              Indexing complete! Opening chat…
            </p>
          </div>
        )}

        {/* Already exists — offer Open or Re-index */}
        {status === 'exists' && existingProject && (
          <div className="card border-yellow-500/30 bg-yellow-500/5 space-y-3">
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-start gap-3">
                <span className="text-2xl">♻️</span>
                <div>
                  <p className="text-yellow-400 font-medium text-sm">
                    Project already indexed
                  </p>
                  <p className="text-gray-400 text-xs mt-1">
                    This codebase was indexed previously and is ready to use.
                    Open it now, or re-index if the source has changed.
                  </p>
                </div>
              </div>
              <button
                onClick={reset}
                className="text-gray-600 hover:text-gray-400 transition-colors shrink-0"
                title="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleOpenExisting}
                className="btn-primary flex-1 text-sm"
              >
                Open Existing
              </button>
              <button
                onClick={handleReindex}
                className="btn-secondary text-sm px-4"
                title="Wipe and re-index from scratch"
              >
                Re-index
              </button>
            </div>
          </div>
        )}

        {/* Error */}
        {status === 'error' && error && (
          <div className="card border-red-800/50 bg-red-950/30 space-y-2">
            <p className="text-red-400 text-sm">⚠️ {error}</p>
            <button
              onClick={reset}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              ← Try again
            </button>
          </div>
        )}
      </div>

      {/* Feature grid */}
      <div className="mt-16 w-full max-w-2xl">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {FEATURES.map(({ Icon, title, desc }) => (
            <div key={title} className="card text-center space-y-2 p-3">
              <Icon className="w-5 h-5 text-indigo-400 mx-auto" />
              <p className="text-white text-xs font-semibold">{title}</p>
              <p className="text-gray-600 text-xs">{desc}</p>
            </div>
          ))}
        </div>
        <p className="text-center text-xs text-gray-800 mt-4">
          Ctrl+K · Chat &nbsp;|&nbsp; Ctrl+M · Memory &nbsp;|&nbsp;
          Ctrl+P · Search projects
        </p>
      </div>
    </div>
  )
}