import { useEffect, useState } from 'react'
import { memoryService } from '@/services/memoryService'
import { ingestService } from '@/services/ingestService'
function ProjectSummary({ projectId, project }) {
  const [summary,   setSummary]   = useState(null)
  const [chromaStats, setChromaStats] = useState(null)

  useEffect(() => {
    if (!projectId) return

    // Load memory for overview
    memoryService.listMemories(projectId).then(data => {
      const overview = (data.memories || []).find(
        m => m.tags?.includes('auto-generated') && m.tags?.includes('overview')
      )
      if (overview) setSummary(overview.content)
    }).catch(() => {})

    // Load ChromaDB stats
    ingestService.getChromaStats(projectId).then(data => {
      setChromaStats(data)
    }).catch(() => {})
  }, [projectId])

  const languages = project?.languages
    ? project.languages.split(',').filter(Boolean)
    : []

  return (
    <div className="w-full max-w-xl space-y-3">
      {/* Project overview */}
      {summary && (
        <div className="card text-left">
          <p className="text-xs text-gray-500 uppercase tracking-wider
                        mb-2 font-medium flex items-center gap-1">
            📋 Project Overview
          </p>
          <p className="text-gray-300 text-sm leading-relaxed">{summary}</p>
        </div>
      )}

      {/* Quick stats */}
      <div className="grid grid-cols-2 gap-2">
        {project && (
          <>
            <div className="card py-3 text-center">
              <p className="text-2xl font-bold text-white">
                {project.file_count}
              </p>
              <p className="text-xs text-gray-500 mt-1">Files indexed</p>
            </div>
            <div className="card py-3 text-center">
              <p className="text-2xl font-bold text-indigo-400">
                {project.chunk_count}
              </p>
              <p className="text-xs text-gray-500 mt-1">Code chunks</p>
            </div>
          </>
        )}
      </div>

      {/* Languages */}
      {languages.length > 0 && (
        <div className="card">
          <p className="text-xs text-gray-500 mb-2">Languages detected</p>
          <div className="flex flex-wrap gap-1.5">
            {languages.map(lang => (
              <span key={lang}
                    className="text-xs px-2 py-1 rounded-full bg-indigo-600/20
                               text-indigo-300 border border-indigo-500/20">
                {lang}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Source URL */}
      {project?.source_url && (
        <div className="card flex items-center gap-2">
          <span className="text-gray-500 text-xs">Source:</span>
          <a href={project.source_url}
             target="_blank"
             rel="noopener noreferrer"
             className="text-xs text-indigo-400 hover:text-indigo-300
                        transition-colors truncate">
            {project.source_url.replace('https://github.com/', '⚡ ')}
          </a>
          {project.branch && (
            <span className="text-xs text-gray-600 shrink-0">
              @ {project.branch}
            </span>
          )}
        </div>
      )}

      {/* Suggested questions */}
      <div className="text-center">
        <p className="text-xs text-gray-600">
          Ask: &ldquo;What does this project do?&rdquo; ·
          &ldquo;Show me the entry point&rdquo; ·
          &ldquo;Explain the architecture&rdquo;
        </p>
      </div>
    </div>
  )
}

export default ProjectSummary