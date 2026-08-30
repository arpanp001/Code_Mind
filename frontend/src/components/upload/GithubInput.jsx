// frontend/src/components/upload/GithubInput.jsx

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Code,
  Loader2,
  Star,
  GitBranch,
  AlertTriangle,
  ChevronDown,
} from 'lucide-react'
import { ingestService } from '@/services/ingestService'

// ── Sub-components ────────────────────────────────────────────────────────────

function RepoPreviewCard({ preview, onClear }) {
  return (
    <div
      className={`rounded-xl p-4 border space-y-3 ${
        preview.is_large
          ? 'border-yellow-500/30 bg-yellow-500/5'
          : 'border-green-500/30 bg-green-500/5'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Code className="w-4 h-4 text-gray-400 shrink-0" />
            <span className="text-white font-semibold text-sm truncate">
              {preview.full_name}
            </span>
          </div>
          {preview.description && (
            <p className="text-gray-400 text-xs mt-1 line-clamp-2">
              {preview.description}
            </p>
          )}
        </div>
        <button
          onClick={onClear}
          className="text-gray-600 hover:text-gray-400 transition-colors text-xs shrink-0"
          title="Clear preview"
        >
          ✕
        </button>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-xs text-gray-500 flex-wrap">
        {preview.language && (
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-indigo-400" />
            {preview.language}
          </span>
        )}
        <span className="flex items-center gap-1">
          <Star className="w-3 h-3" />
          {(preview.stars || 0).toLocaleString()}
        </span>
        <span className="flex items-center gap-1">
          <GitBranch className="w-3 h-3" />
          {preview.default_branch}
        </span>
        <span>{preview.size_mb} MB</span>
      </div>

      {/* Large repo warning */}
      {preview.is_large && (
        <div className="flex items-start gap-2 bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-2.5">
          <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-yellow-400 text-xs font-medium">
              Large repository ({preview.size_mb} MB)
            </p>
            <p className="text-yellow-400/70 text-xs mt-0.5">
              Estimated indexing time: ~{preview.estimated_minutes} min. Only
              source code will be indexed (tests/docs skipped).
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function BranchSelector({ url, value, onChange, defaultBranch }) {
  const [branches, setBranches] = useState([defaultBranch || 'main'])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  const loadBranches = useCallback(async () => {
    if (!url || branches.length > 1) return
    try {
      setLoading(true)
      const data = await ingestService.listBranches(url)
      if (data?.branches?.length) setBranches(data.branches)
    } catch (_) {
      // Keep default
    } finally {
      setLoading(false)
    }
  }, [url, branches.length])

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleToggle = () => {
    if (!open) loadBranches()
    setOpen(!open)
  }

  return (
    <div ref={ref} className="relative">
      <label className="text-xs text-gray-500 mb-1 block">Branch</label>
      <button
        type="button"
        onClick={handleToggle}
        className="input-field flex items-center justify-between text-left"
      >
        <span className="flex items-center gap-2">
          <GitBranch className="w-3.5 h-3.5 text-gray-500" />
          <span>{value || defaultBranch || 'main'}</span>
          {value === defaultBranch && (
            <span className="text-xs text-gray-600">(default)</span>
          )}
        </span>
        {loading ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-500" />
        ) : (
          <ChevronDown
            className={`w-3.5 h-3.5 text-gray-500 transition-transform ${
              open ? 'rotate-180' : ''
            }`}
          />
        )}
      </button>

      {open && (
        <div className="absolute top-full mt-1 left-0 right-0 z-50 bg-gray-800 border border-gray-700 rounded-xl shadow-xl overflow-hidden max-h-48 overflow-y-auto">
          {branches.map((b) => (
            <button
              key={b}
              type="button"
              onClick={() => {
                onChange(b)
                setOpen(false)
              }}
              className={`w-full text-left px-3 py-2 text-sm flex items-center gap-2 hover:bg-gray-700 transition-colors ${
                b === value ? 'text-indigo-400' : 'text-gray-300'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5 text-gray-600" />
              {b}
              {b === defaultBranch && (
                <span className="text-xs text-gray-600 ml-auto">default</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function GithubInput({ onSubmit, isLoading }) {
  const [url, setUrl] = useState('')
  const [branch, setBranch] = useState('main')
  const [preview, setPreview] = useState(null)
  const [previewing, setPreviewing] = useState(false)
  const [err, setErr] = useState('')
  const [errIsWarn, setErrIsWarn] = useState(false)

  const debounceRef = useRef(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)

    if (!url || !url.includes('github.com/')) {
      setPreview(null)
      setErr('')
      setErrIsWarn(false)
      return
    }

    const parts = url.replace('https://github.com/', '').split('/').filter(Boolean)
    if (parts.length < 2) {
      setPreview(null)
      return
    }

    debounceRef.current = setTimeout(() => {
      fetchPreview(url)
    }, 800)

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [url])

  const fetchPreview = async (rawUrl) => {
    try {
      setPreviewing(true)
      setErr('')
      setErrIsWarn(false)

      const data = await ingestService.previewGithubRepo(rawUrl)
      setPreview(data)
      if (data.default_branch) setBranch(data.default_branch)
    } catch (e) {
      setPreview(null)
      const status = e.response?.status
      const detail = e.response?.data?.detail || ''
      const isTimeout =
        e.code === 'ECONNABORTED' ||
        e.message?.includes('timeout') ||
        status === 504

      if (isTimeout) {
        setErr(
          'GitHub API timed out. Preview unavailable — you can still click "Index Repository".'
        )
        setErrIsWarn(true)
      } else if (status === 404) {
        setErr('Repository not found. Check the URL and make sure it\'s public.')
      } else if (status === 403) {
        setErr('Access denied. This may be a private repository.')
      } else if (detail) {
        setErr(detail)
      } else {
        setErr('Could not fetch repository info. Check the URL.')
      }
    } finally {
      setPreviewing(false)
    }
  }

  const handleSubmit = (e) => {
    if (e) e.preventDefault()

    if (!url.trim()) {
      setErr('Repository URL is required')
      return
    }
    if (!url.includes('github.com')) {
      setErr('Must be a GitHub URL[](https://github.com/owner/repo)')
      return
    }

    setErr('')
    onSubmit(url.trim(), branch)
  }

  const handleClearPreview = () => {
    setPreview(null)
    setUrl('')
    setErr('')
    setBranch('main')
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gray-800 flex items-center justify-center shrink-0">
          <Code className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="font-semibold text-white text-sm">GitHub Repository</h3>
          <p className="text-xs text-gray-500">
            Public repositories · Auto-preview on paste
          </p>
        </div>
      </div>

      {/* URL Input */}
      <div>
        <label className="text-xs text-gray-500 mb-1 block">Repository URL</label>
        <div className="relative">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSubmit()
            }}
            placeholder="https://github.com/owner/repository"
            className="input-field font-mono pr-8"
            disabled={isLoading}
          />
          {previewing && (
            <Loader2 className="absolute right-2.5 top-2.5 w-4 h-4 animate-spin text-gray-500" />
          )}
        </div>

        {err && (
          <div
            className={`flex items-start gap-2 text-xs mt-2 rounded-lg p-2.5 ${
              errIsWarn
                ? 'bg-yellow-500/10 border border-yellow-500/20 text-yellow-400'
                : 'bg-red-500/10 border border-red-500/20 text-red-400'
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>{err}</span>
          </div>
        )}
      </div>

      {/* Preview Card */}
      {preview && !isLoading && (
        <RepoPreviewCard preview={preview} onClear={handleClearPreview} />
      )}

      {/* Branch Selector */}
      {preview && !isLoading && (
        <BranchSelector
          url={url}
          value={branch}
          onChange={setBranch}
          defaultBranch={preview.default_branch}
        />
      )}

      {/* Submit Button */}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={isLoading || !url.trim() || previewing}
        className="btn-primary w-full flex items-center justify-center gap-2"
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" /> Cloning &amp; indexing…
          </>
        ) : previewing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" /> Checking repository…
          </>
        ) : (
          <>
            <Code className="w-4 h-4" /> Index Repository
          </>
        )}
      </button>

      <p className="text-xs text-gray-700 text-center">
        Paste a GitHub URL to auto-preview · Supports public repos
      </p>
    </div>
  )
}