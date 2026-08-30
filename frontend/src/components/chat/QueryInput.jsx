// frontend/src/components/chat/QueryInput.jsx
import { useState, useRef, useEffect } from 'react'
import { Send, Lightbulb } from 'lucide-react'

const EXAMPLES = [
  'Where is login implemented?',
  'Explain the authentication flow',
  'Which file connects to the database?',
  'What design patterns are used?',
  'Where are API routes defined?',
]

export default function QueryInput({ onSend, isLoading, disabled }) {
  const [value,    setValue]    = useState('')
  const [showHints, setShowHints] = useState(false)
  const taRef = useRef(null)

  useEffect(() => {
    const ta = taRef.current; if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 140)}px`
  }, [value])

  const send = () => {
    if (!value.trim() || isLoading || disabled) return
    onSend(value.trim()); setValue(''); setShowHints(false)
  }

  return (
    <div className="relative">
      {showHints && !value && (
        <div className="absolute bottom-full mb-2 inset-x-0 bg-gray-800
                        border border-gray-700 rounded-xl p-2 shadow-xl z-10">
          <p className="text-xs text-gray-500 px-2 pb-1">Example questions</p>
          {EXAMPLES.map((q, i) => (
            <button
              key={i}
              onClick={() => { setValue(q); setShowHints(false) }}
              className="w-full text-left text-xs text-gray-300
                         hover:bg-gray-700 hover:text-white px-3 py-1.5
                         rounded-lg transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2 bg-gray-800 border border-gray-700
                      rounded-2xl px-3 py-2.5
                      focus-within:border-indigo-500 transition-colors">
        <button
          onClick={() => setShowHints(!showHints)}
          className="text-gray-600 hover:text-yellow-400 transition-colors
                     shrink-0 mb-0.5"
        >
          <Lightbulb className="w-4 h-4" />
        </button>

        <textarea
          ref={taRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          onFocus={() => setShowHints(false)}
          placeholder={disabled ? 'Select a project to start chatting…'
                                : 'Ask anything about the codebase…'}
          disabled={isLoading || disabled}
          rows={1}
          className="flex-1 bg-transparent text-gray-100 placeholder-gray-600
                     text-sm resize-none outline-none max-h-36 leading-relaxed"
        />

        <button
          onClick={send}
          disabled={!value.trim() || isLoading || disabled}
          className="shrink-0 mb-0.5 w-7 h-7 rounded-lg bg-indigo-600
                     hover:bg-indigo-700 disabled:opacity-30
                     flex items-center justify-center transition-colors"
        >
          <Send className="w-3.5 h-3.5 text-white" />
        </button>
      </div>

      <p className="text-center text-xs text-gray-700 mt-1">
        Enter to send · Shift+Enter for new line
      </p>
    </div>
  )
}