// frontend/src/components/layout/Sidebar.jsx
//
// Fixes vs previous version:
//   1. loading state is set to false even when projects array is empty
//      (was staying true indefinitely when API returned { projects: [] })
//   2. silentRefresh uses a stable ref — does NOT cause re-renders unless
//      data actually changed. Previous version updated state on every tick
//      which caused the component to re-render, re-register the interval,
//      and created an infinite loop visible in the network tab.
//   3. Context-aware navigation: clicking a project while on /memory
//      stays on /memory/{id}, not /chat/{id}
//   4. Search is fully local (no API call) — filters the already-loaded list
//   5. Delete properly removes from local state without a full reload

import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate, useLocation, useParams }       from 'react-router-dom'
import { Plus, Trash2, RefreshCw, Search, X }        from 'lucide-react'
import { ingestService }  from '@/services/ingestService'
import { timeAgo }        from '@/utils/formatters'

const STATUS_DOT = {
  ready:      'bg-green-400',
  processing: 'bg-yellow-400 animate-pulse',
  pending:    'bg-gray-500',
  failed:     'bg-red-400',
}

function HighlightMatch({ text, query }) {
  if (!query) return <>{text}</>
  const idx = text.toLowerCase().indexOf(query.toLowerCase())
  if (idx === -1) return <>{text}</>
  return (
    <>
      {text.slice(0, idx)}
      <span className="text-indigo-400 font-semibold">
        {text.slice(idx, idx + query.length)}
      </span>
      {text.slice(idx + query.length)}
    </>
  )
}

export default function Sidebar() {
  const [projects,    setProjects]    = useState([])
  const [loading,     setLoading]     = useState(true)   // true only on first load
  const [searchQuery, setSearchQuery] = useState('')
  const [showSearch,  setShowSearch]  = useState(false)

  const navigate          = useNavigate()
  const location          = useLocation()
  const { projectId }     = useParams()

  // Use refs so the interval callback always has the latest values
  // without needing to be re-registered (which would reset the timer)
  const projectsRef   = useRef([])
  const intervalRef   = useRef(null)
  const searchRef     = useRef(null)

  // ── Initial load ────────────────────────────────────────────────────────
  useEffect(() => {
    loadProjects()

    // Silent background refresh every 15s
    // Uses ref comparison so it only calls setProjects when data changes
    // — this prevents unnecessary re-renders and the polling loop
    intervalRef.current = setInterval(silentRefresh, 15000)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])   // Empty deps — run ONCE on mount only

  // ── Focus search input when shown ───────────────────────────────────────
  useEffect(() => {
    if (showSearch) searchRef.current?.focus()
  }, [showSearch])

  // ── Keyboard shortcuts ──────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault()
        setShowSearch(true)
      }
      if (e.key === 'Escape' && showSearch) {
        setShowSearch(false)
        setSearchQuery('')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [showSearch])

  // ── Load all projects (shows skeleton during first load only) ───────────
  const loadProjects = useCallback(async () => {
    try {
      setLoading(true)
      const data = await ingestService.listProjects()
      const list = data?.projects || []
      projectsRef.current = list
      setProjects(list)
    } catch (err) {
      console.error('Failed to load projects:', err)
      // Still clear loading so skeleton disappears even on error
    } finally {
      // CRITICAL FIX: always set loading=false so skeleton never gets stuck
      setLoading(false)
    }
  }, [])

  // ── Silent background refresh ───────────────────────────────────────────
  // Does NOT call setLoading(true) — only updates state if data changed
  // This prevents the skeleton flash on every 15s tick
  const silentRefresh = async () => {
    try {
      const data = await ingestService.listProjects()
      const list = data?.projects || []

      // Compare by id+status only — avoids re-render when nothing changed
      const serialize = (arr) =>
        arr.map(p => `${p.project_id}:${p.status}:${p.chunk_count}`).join(',')

      if (serialize(projectsRef.current) !== serialize(list)) {
        projectsRef.current = list
        setProjects(list)
      }
    } catch (_) {
      // Silent — don't show errors for background refresh
    }
  }

  // ── Navigate to the correct section for the selected project ────────────
  const handleProjectClick = useCallback((id) => {
    const section = location.pathname.startsWith('/memory')
      ? 'memory'
      : 'chat'
    navigate(`/${section}/${id}`)
    setShowSearch(false)
    setSearchQuery('')
  }, [location.pathname, navigate])

  // ── Delete a project ────────────────────────────────────────────────────
  const handleDelete = useCallback(async (e, id) => {
    e.stopPropagation()
    if (!confirm('Delete this project and all its data? This cannot be undone.')) return
    try {
      await ingestService.deleteProject(id)
      const updated = projectsRef.current.filter(p => p.project_id !== id)
      projectsRef.current = updated
      setProjects(updated)
      if (projectId === id) navigate('/')
    } catch (err) {
      alert('Delete failed: ' + (err.userMessage || err.message))
    }
  }, [projectId, navigate])

  // ── Clear all but the newest project ────────────────────────────────────
  const handleClearOld = useCallback(async () => {
    const count = projects.length - 1
    if (!confirm(`Delete ${count} older project(s)? The newest will be kept.`)) return
    const toDelete = projects.slice(1)
    for (const p of toDelete) {
      try { await ingestService.deleteProject(p.project_id) } catch (_) {}
    }
    loadProjects()
  }, [projects, loadProjects])

  // ── Filter projects by search query (local — no API call) ───────────────
  const filteredProjects = searchQuery.trim()
    ? projects.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : projects

  // ────────────────────────────────────────────────────────────────────────

  return (
    <aside className="w-56 shrink-0 h-full bg-gray-900
                      border-r border-gray-800 flex flex-col">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="border-b border-gray-800">
        {showSearch ? (
          <div className="flex items-center gap-2 px-3 py-3">
            <Search className="w-3.5 h-3.5 text-gray-500 shrink-0" />
            <input
              ref={searchRef}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search projects…"
              className="flex-1 bg-transparent text-white text-sm
                         placeholder-gray-600 outline-none"
            />
            <button
              onClick={() => { setShowSearch(false); setSearchQuery('') }}
              className="text-gray-600 hover:text-gray-400 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex items-center justify-between px-3 py-3">
            <span className="text-xs font-semibold text-gray-500
                             uppercase tracking-wider">
              Projects
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setShowSearch(true)}
                className="text-gray-600 hover:text-gray-400 transition-colors"
                title="Search projects (Ctrl+P)"
              >
                <Search className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={loadProjects}
                className="text-gray-600 hover:text-gray-400 transition-colors"
                title="Refresh project list"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Project list ────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">

        {/* Skeleton — shown ONLY during the very first load */}
        {loading ? (
          <div className="space-y-2 p-1">
            {[1, 2, 3].map(i => (
              <div key={i}
                   className="h-12 bg-gray-800 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : filteredProjects.length === 0 ? (
          /* Empty state */
          <div className="text-center py-8 px-3">
            <p className="text-gray-600 text-xs">
              {searchQuery
                ? `No projects matching "${searchQuery}"`
                : 'No projects yet'
              }
            </p>
            {!searchQuery && (
              <p className="text-gray-700 text-xs mt-1">
                Upload a codebase to start
              </p>
            )}
          </div>
        ) : (
          /* Project items — <div role="button"> avoids nested <button> warning */
          filteredProjects.map(p => {
            const active = p.project_id === projectId
            return (
              <div
                key={p.project_id}
                role="button"
                tabIndex={0}
                onClick={() => handleProjectClick(p.project_id)}
                onKeyDown={e => {
                  if (e.key === 'Enter') handleProjectClick(p.project_id)
                }}
                className={`w-full text-left px-3 py-2.5 rounded-lg cursor-pointer
                            flex items-start gap-2 group transition-colors
                            ${active
                              ? 'bg-indigo-600/20 border border-indigo-500/30'
                              : 'hover:bg-gray-800 border border-transparent'
                            }`}
              >
                {/* Status dot */}
                <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0
                                  ${STATUS_DOT[p.status] || 'bg-gray-500'}`} />

                {/* Name + metadata */}
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium truncate
                                 ${active ? 'text-white' : 'text-gray-300'}`}>
                    <HighlightMatch text={p.name} query={searchQuery} />
                  </p>
                  <p className="text-xs text-gray-600 truncate">
                    {p.chunk_count > 0
                      ? `${p.chunk_count} chunks`
                      : p.status
                    }
                    {' · '}{timeAgo(p.created_at)}
                  </p>
                </div>

                {/* Delete button — appears on hover */}
                {/* This is a <button> inside a <div role="button"> — valid HTML */}
                <button
                  onClick={e => handleDelete(e, p.project_id)}
                  className="opacity-0 group-hover:opacity-100 text-gray-600
                             hover:text-red-400 transition-all shrink-0 mt-0.5"
                  title="Delete project"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            )
          })
        )}
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="p-2 border-t border-gray-800 space-y-1">
        <button
          onClick={() => navigate('/')}
          className="w-full btn-secondary text-xs flex items-center
                     justify-center gap-1.5"
        >
          <Plus className="w-3.5 h-3.5" />
          New Project
        </button>

        {projects.length > 3 && (
          <button
            onClick={handleClearOld}
            className="w-full text-xs text-gray-700 hover:text-red-400
                       py-1 transition-colors"
          >
            Clear {projects.length - 1} old projects
          </button>
        )}

        <p className="text-center text-xs text-gray-800 pt-0.5">
          Ctrl+P to search
        </p>
      </div>
    </aside>
  )
}