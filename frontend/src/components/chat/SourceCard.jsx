// frontend/src/components/chat/SourceCard.jsx
// COMPLETE REPLACEMENT

import { useState } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { ChevronDown, FileCode, Layers } from 'lucide-react'
import { getLanguageColor } from '@/utils/formatters'
import { Copy, Check } from 'lucide-react'

function RelevanceBadge({ score }) {
  const pct   = Math.round((score || 0) * 100)
  const color =
    pct >= 80 ? 'text-green-400  bg-green-400/10  border-green-400/30'  :
    pct >= 60 ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30' :
                'text-gray-400   bg-gray-400/10   border-gray-700'
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full
                      border shrink-0 ${color}`}>
      {pct}%
    </span>
  )
}

export default function SourceCard({ source, index }) {
  const [open,         setOpen]         = useState(false)
  const [activeRange,  setActiveRange]  = useState(0)
  const [copied, setCopied] = useState(false)


  // Multi-range support (grouped chunks from same file)
  const ranges = source.all_ranges || [{
    start: source.start_line,
    end:   source.end_line,
    code:  source.code,
    score: source.relevance_score,
  }]

  const currentRange = ranges[activeRange]
  const hasMultiple  = ranges.length > 1
  const handleCopy = async (e) => {
  e.stopPropagation()

  try {
    await navigator.clipboard.writeText(
      currentRange?.code || source.code || ''
    )

    setCopied(true)

    setTimeout(() => {
      setCopied(false)
    }, 2000)
  } catch (_) {}
}


  return (
    <div className="border border-gray-700/60 rounded-xl overflow-hidden
                    bg-gray-800/40">

      {/* Header row */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen(!open)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            setOpen(!open)
          }
        }}
        className="w-full text-left px-3 py-2.5 flex items-center gap-2.5
                  hover:bg-gray-700/30 transition-colors cursor-pointer"
      >
        <span className="text-gray-600 font-mono text-xs w-4 shrink-0
                         text-center">
          {index + 1}
        </span>

        <FileCode className={`w-3.5 h-3.5 shrink-0
                              ${getLanguageColor(source.language)}`} />

        <span className="font-mono text-xs text-indigo-300 flex-1 truncate">
          {source.file_path}
        </span>

        {/* Multi-range indicator */}
        {hasMultiple && (
          <span className="flex items-center gap-1 text-xs text-gray-500 shrink-0">
            <Layers className="w-3 h-3" />
            {ranges.length} sections
          </span>
        )}

        {/* Line range */}
        {currentRange?.start && !hasMultiple && (
          <span className="text-xs text-gray-600 shrink-0 hidden sm:block">
            L{currentRange.start}–{currentRange.end}
          </span>
        )}

        {/* Chunk type */}
        {source.chunk_type && source.chunk_type !== 'block' && (
          <span className="text-xs text-gray-500 bg-gray-800 px-1.5 py-0.5
                           rounded shrink-0 hidden sm:block">
            {source.chunk_type}
          </span>
        )}

        <RelevanceBadge score={source.relevance_score} />

         <button
          onClick={handleCopy}
          className="text-gray-600 hover:text-gray-300 transition-colors
                     shrink-0"
          title="Copy code"
        >
          {copied
            ? <Check className="w-3.5 h-3.5 text-green-400" />
            : <Copy  className="w-3.5 h-3.5" />
          }
        </button>

        <ChevronDown className={`w-3.5 h-3.5 text-gray-500 shrink-0
                                  transition-transform duration-200
                                  ${open ? 'rotate-180' : ''}`} />
      </div>

      {/* Expandable content */}
      {open && (
        <div className="border-t border-gray-700/60">
          {/* Range tabs when multiple sections */}
          {hasMultiple && (
            <div className="flex gap-1 px-3 pt-2 pb-1 overflow-x-auto">
              {ranges.map((r, i) => (
                <button
                  key={i}
                  onClick={() => setActiveRange(i)}
                  className={`text-xs px-2.5 py-1 rounded-md shrink-0
                               transition-colors
                               ${activeRange === i
                                 ? 'bg-indigo-600 text-white'
                                 : 'bg-gray-700 text-gray-400 hover:text-white'
                               }`}
                >
                  {r.start ? `L${r.start}–${r.end}` : `Section ${i + 1}`}
                  <span className="ml-1 opacity-60">
                    {Math.round((r.score || 0) * 100)}%
                  </span>
                </button>
              ))}
            </div>
          )}

          <SyntaxHighlighter
            language={source.language || 'text'}
            style={vscDarkPlus}
            customStyle={{
              margin: 0, borderRadius: 0,
              fontSize: '11px', maxHeight: '280px',
            }}
            showLineNumbers={!!currentRange?.start}
            startingLineNumber={currentRange?.start || 1}
          >
            {currentRange?.code || source.code || ''}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  )
}