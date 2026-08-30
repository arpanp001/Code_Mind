// frontend/src/pages/MemoryPage.jsx
// COMPLETE REPLACEMENT

import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Plus, Search, Trash2, X, Loader2, Brain } from 'lucide-react'
import Sidebar from '@/components/layout/Sidebar'
import { memoryService } from '@/services/memoryService'
import { ingestService } from '@/services/ingestService'
import { getMemoryTypeInfo, timeAgo } from '@/utils/formatters'

const TYPES = [
  { value: 'architecture_decision', label: 'Architecture Decision' },
  { value: 'bug_fix',               label: 'Bug Fix'               },
  { value: 'note',                  label: 'Note'                  },
]

const EMPTY_FORM = {
  content:     '',
  memory_type: 'architecture_decision',
  title:       '',
  tags:        '',
}

export default function MemoryPage() {
  const { projectId } = useParams()
  const navigate      = useNavigate()

  const [project,  setProject]  = useState(null)
  const [memories, setMemories] = useState([])
  const [loading,  setLoading]  = useState(false)
  const [filter,   setFilter]   = useState('all')
  const [search,   setSearch]   = useState('')
  const [showForm, setShowForm] = useState(false)
  const [saving,   setSaving]   = useState(false)
  const [saveErr,  setSaveErr]  = useState('')
  const [form,     setForm]     = useState(EMPTY_FORM)

  // Load project info whenever projectId changes
  useEffect(() => {
    if (!projectId) {
      setProject(null)
      setMemories([])
      return
    }
    loadProject()
    loadMemories()
  }, [projectId])

  const loadProject = async () => {
    try {
      const data = await ingestService.getStatus(projectId)
      setProject(data)
    } catch (_) {
      setProject(null)
    }
  }

  const loadMemories = useCallback(async () => {
    if (!projectId) return
    try {
      setLoading(true)
      const data = await memoryService.listMemories(projectId)
      setMemories(data.memories || [])
    } catch (_) {
      setMemories([])
    } finally {
      setLoading(false)
    }
  }, [projectId])

  const handleSave = async (e) => {
    e.preventDefault()
    setSaveErr('')

    if (!projectId) {
      setSaveErr('No project selected. Select a project from the sidebar.')
      return
    }
    if (!form.content.trim()) {
      setSaveErr('Content is required.')
      return
    }

    try {
      setSaving(true)
      await memoryService.addMemory(
        projectId,
        form.content.trim(),
        form.memory_type,
        form.tags.split(',').map(t => t.trim()).filter(Boolean),
        form.title.trim(),
      )
      // Reset form and close
      setForm(EMPTY_FORM)
      setShowForm(false)
      // Reload the list
      await loadMemories()
    } catch (err) {
      setSaveErr(err.userMessage || err.message || 'Failed to save memory')
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (memoryId) => {
    if (!confirm('Delete this memory?')) return
    try {
      await memoryService.deleteMemory(memoryId, projectId)
      setMemories(prev => prev.filter(m => m.memory_id !== memoryId))
    } catch (err) {
      alert('Failed to delete: ' + (err.userMessage || err.message))
    }
  }

  const handleSearch = async () => {
    if (!search.trim()) {
      loadMemories()
      return
    }
    try {
      setLoading(true)
      const data = await memoryService.searchMemories(projectId, search)
      setMemories(data.memories || [])
    } catch (_) {
    } finally {
      setLoading(false)
    }
  }

  const handleClearSearch = () => {
    setSearch('')
    loadMemories()
  }

  const filtered = filter === 'all'
    ? memories
    : memories.filter(m => m.memory_type === filter)

  // ── No project selected state ──────────────────────────────────────────────
  if (!projectId) {
    return (
      <div className="flex h-screen pt-14">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center
                        text-center space-y-4 px-6">
          <Brain className="w-12 h-12 text-gray-700" />
          <h2 className="text-white font-semibold text-lg">Project Memory</h2>
          <p className="text-gray-500 text-sm max-w-xs">
            Select a project from the sidebar to view and manage its
            architecture decisions, bug fixes, and notes.
          </p>
          <p className="text-gray-700 text-xs">
            Or upload a new codebase to get started.
          </p>
        </div>
      </div>
    )
  }

  // ── Main memory page ───────────────────────────────────────────────────────
  return (
    <div className="flex h-screen pt-14">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Header */}
        <div className="border-b border-gray-800 px-6 py-4 shrink-0
                        flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h1 className="text-white font-semibold">Project Memory</h1>
            <p className="text-gray-500 text-sm truncate">
              {project
                ? `${project.name} · ${memories.length} memor${memories.length === 1 ? 'y' : 'ies'}`
                : 'Loading…'
              }
            </p>
          </div>

          <button
            onClick={() => {
              setShowForm(prev => !prev)
              setSaveErr('')
            }}
            className="btn-primary text-sm flex items-center gap-1.5 shrink-0"
          >
            {showForm
              ? <><X className="w-4 h-4" /> Cancel</>
              : <><Plus className="w-4 h-4" /> Add Memory</>
            }
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">

          {/* ── Add Memory Form ─────────────────────────────────────────── */}
          {showForm && (
            <div className="card border border-indigo-500/30 space-y-4">
              <h2 className="text-white font-medium text-sm">New Memory</h2>

              {/* Memory Type */}
              <div>
                <label className="text-xs text-gray-500 mb-1 block">
                  Memory Type *
                </label>
                <select
                  value={form.memory_type}
                  onChange={e => setForm({ ...form, memory_type: e.target.value })}
                  className="input-field"
                  disabled={saving}
                >
                  {TYPES.map(t => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>

              {/* Title */}
              <div>
                <label className="text-xs text-gray-500 mb-1 block">
                  Title (optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g. JWT Token Strategy"
                  value={form.title}
                  onChange={e => setForm({ ...form, title: e.target.value })}
                  className="input-field"
                  disabled={saving}
                />
              </div>

              {/* Content */}
              <div>
                <label className="text-xs text-gray-500 mb-1 block">
                  Content *
                </label>
                <textarea
                  placeholder="Describe the decision, bug fix, or note in detail…"
                  value={form.content}
                  onChange={e => setForm({ ...form, content: e.target.value })}
                  rows={5}
                  className="input-field resize-none"
                  disabled={saving}
                />
              </div>

              {/* Tags */}
              <div>
                <label className="text-xs text-gray-500 mb-1 block">
                  Tags (comma separated)
                </label>
                <input
                  type="text"
                  placeholder="jwt, auth, security"
                  value={form.tags}
                  onChange={e => setForm({ ...form, tags: e.target.value })}
                  className="input-field"
                  disabled={saving}
                />
              </div>

              {/* Error */}
              {saveErr && (
                <div className="bg-red-950/40 border border-red-800/50
                                rounded-lg p-3">
                  <p className="text-red-400 text-xs">{saveErr}</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={handleSave}
                  disabled={!form.content.trim() || saving}
                  className="btn-primary flex-1 flex items-center
                             justify-center gap-2"
                >
                  {saving
                    ? <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
                    : 'Save Memory'
                  }
                </button>
                <button
                  onClick={() => {
                    setForm(EMPTY_FORM)
                    setShowForm(false)
                    setSaveErr('')
                  }}
                  className="btn-secondary px-4"
                  disabled={saving}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* ── Search ──────────────────────────────────────────────────── */}
          <div className="flex gap-2">
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="Semantic search memories…"
              className="input-field flex-1"
            />
            <button onClick={handleSearch}
                    className="btn-secondary px-3"
                    title="Search">
              <Search className="w-4 h-4" />
            </button>
            {search && (
              <button onClick={handleClearSearch}
                      className="btn-ghost px-3"
                      title="Clear search">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* ── Filter Tabs ──────────────────────────────────────────────── */}
          <div className="flex gap-2 flex-wrap">
            {['all', ...TYPES.map(t => t.value)].map(type => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium
                            capitalize transition-colors
                            ${filter === type
                              ? 'bg-indigo-600 text-white'
                              : 'bg-gray-800 text-gray-400 hover:text-white'
                            }`}
              >
                {type === 'all' ? 'All' : type.replace(/_/g, ' ')}
              </button>
            ))}
          </div>

          {/* ── Memory List ──────────────────────────────────────────────── */}
          {loading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i}
                     className="h-24 bg-gray-800 rounded-xl animate-pulse" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-4xl mb-3">🧠</div>
              <p className="text-gray-500 text-sm">
                {search ? 'No memories matched your search.' : 'No memories yet.'}
              </p>
              {!search && (
                <p className="text-gray-700 text-xs mt-2">
                  Click "+ Add Memory" above to store your first memory.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {filtered.map(mem => {
                const info = getMemoryTypeInfo(mem.memory_type)
                return (
                  <div
                    key={mem.memory_id}
                    className="card hover:border-gray-700 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span>{info.emoji}</span>
                        <span className={`text-xs font-medium px-2 py-0.5
                                         rounded-full border ${info.color}`}>
                          {info.label}
                        </span>
                        {mem.title && (
                          <span className="text-white font-medium text-sm">
                            {mem.title}
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => handleDelete(mem.memory_id)}
                        className="text-gray-700 hover:text-red-400
                                   transition-colors shrink-0"
                        title="Delete memory"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <p className="text-gray-300 text-sm mt-2 leading-relaxed">
                      {mem.content}
                    </p>

                    <div className="flex items-center justify-between mt-3 gap-2">
                      {mem.tags?.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {mem.tags.map(tag => (
                            <span
                              key={tag}
                              className="text-xs px-2 py-0.5 rounded-full
                                         bg-gray-800 text-gray-500"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>
                      )}
                      <div className="flex items-center gap-3 ml-auto">
                        {mem.relevance_score && (
                          <span className="text-xs text-indigo-400">
                            {Math.round(mem.relevance_score * 100)}% match
                          </span>
                        )}
                        <span className="text-xs text-gray-700">
                          {timeAgo(mem.created_at)}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}