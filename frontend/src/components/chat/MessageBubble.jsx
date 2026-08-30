// frontend/src/components/chat/MessageBubble.jsx
// COMPLETE REPLACEMENT

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Brain, User, AlertCircle } from 'lucide-react'

function CodeContent({ content }) {
  const parts = content.split(/(```[\w]*\n[\s\S]*?```)/g)
  return (
    <div className="space-y-2 text-sm leading-relaxed">
      {parts.map((part, i) => {
        const m = part.match(/```(\w*)\n([\s\S]*?)```/)
        if (m) {
          const [, lang, code] = m
          return (
            <SyntaxHighlighter
              key={i}
              language={lang || 'text'}
              style={vscDarkPlus}
              customStyle={{ borderRadius: '8px', fontSize: '12px', margin: 0 }}
            >
              {code.trim()}
            </SyntaxHighlighter>
          )
        }
        // Render inline `code` spans
        const inlineParts = part.split(/(`[^`]+`)/g)
        return part ? (
          <p key={i} className="whitespace-pre-wrap">
            {inlineParts.map((ip, j) =>
              ip.startsWith('`') && ip.endsWith('`') && ip.length > 2
                ? <code key={j}
                        className="bg-gray-700/60 text-indigo-300 px-1 py-0.5
                                   rounded text-xs font-mono">
                    {ip.slice(1, -1)}
                  </code>
                : ip
            )}
          </p>
        ) : null
      })}
    </div>
  )
}

function MessageMeta({ message, isUser }) {
  const time = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className={`flex items-center gap-2 mt-2 text-xs opacity-40
                     ${isUser ? 'justify-end' : 'justify-start'}`}>
      <span>{time}</span>
      {/* Token count — only shown ONCE for assistant messages */}
      {!isUser && message.tokens > 0 && (
        <span>{message.tokens} tokens</span>
      )}
      {/* Memory injection indicator */}
      {!isUser && message.memoriesUsed > 0 && (
        <span className="opacity-100 text-indigo-400 font-medium">
          🧠 {message.memoriesUsed}
        </span>
      )}
    </div>
  )
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const isErr  = message.role === 'error'
  const isAI   = message.role === 'assistant'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>

      {/* Avatar */}
      <div className={`w-7 h-7 rounded-full shrink-0 flex items-center
                       justify-center mt-0.5
                       ${isUser
                         ? 'bg-indigo-600'
                         : isErr
                           ? 'bg-red-900'
                           : 'bg-gray-800 border border-gray-700'}`}>
        {isUser
          ? <User        className="w-3.5 h-3.5 text-white" />
          : isErr
            ? <AlertCircle className="w-3.5 h-3.5 text-red-400" />
            : <Brain       className="w-3.5 h-3.5 text-indigo-400" />
        }
      </div>

      {/* Bubble */}
      <div className={`max-w-[82%] rounded-2xl px-3 py-2.5
                       ${isUser
                         ? 'bg-indigo-600 text-white rounded-tr-sm'
                         : isErr
                           ? 'bg-red-950 text-red-400 border border-red-900/60 rounded-tl-sm'
                           : 'bg-gray-800/80 text-gray-100 rounded-tl-sm'}`}>
        {isAI
          ? <CodeContent content={message.content} />
          : <p className="text-sm leading-relaxed">{message.content}</p>
        }

        {/* Single metadata row — no duplication */}
        <MessageMeta message={message} isUser={isUser} />
      </div>
    </div>
  )
}