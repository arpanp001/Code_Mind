// frontend/src/utils/exportChat.js

export function exportChatAsMarkdown(messages, projectName) {
    const lines = [
        `# CodeMind Chat Export`,
        `**Project:** ${projectName}`,
        `**Exported:** ${new Date().toLocaleString()}`,
        `**Messages:** ${messages.filter(m => m.role !== 'error').length}`,
        '',
        '---',
        '',
    ]

    for (const msg of messages) {
        if (msg.role === 'error') continue

        const time = new Date(msg.timestamp).toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit'
        })

        if (msg.role === 'user') {
            lines.push(`## 🙋 User *(${time})*`)
            lines.push('')
            lines.push(msg.content)
            lines.push('')
        } else if (msg.role === 'assistant') {
            lines.push(`## 🧠 CodeMind *(${time})*`)
            lines.push('')
            lines.push(msg.content)
            if (msg.tokens) {
                lines.push('')
                lines.push(`*${msg.tokens} tokens used*`)
            }
            if (msg.sources?.length) {
                lines.push('')
                lines.push(`**Sources:** ${msg.sources.map(s => `\`${s.file_path}\``).join(', ')}`)
            }
            lines.push('')
        }

        lines.push('---')
        lines.push('')
    }

    return lines.join('\n')
}

export function downloadMarkdown(content, filename) {
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
}

export function exportChatAsHTML(messages, projectName) {
    const rows = messages
        .filter(m => m.role !== 'error')
        .map(msg => {
            const isUser = msg.role === 'user'
            const time = new Date(msg.timestamp).toLocaleTimeString([], {
                hour: '2-digit', minute: '2-digit'
            })
            const bgColor = isUser ? '#4f46e5' : '#1f2937'
            const align = isUser ? 'right' : 'left'

            return `
        <div style="margin:16px 0; text-align:${align}">
          <div style="display:inline-block; max-width:80%; background:${bgColor};
                      color:#fff; padding:12px 16px; border-radius:12px;
                      text-align:left; font-family:sans-serif; font-size:14px;
                      line-height:1.6; white-space:pre-wrap;">
            ${msg.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
          </div>
          <div style="font-size:11px; color:#6b7280; margin-top:4px">
            ${isUser ? 'You' : 'CodeMind'} · ${time}
            ${msg.tokens ? ` · ${msg.tokens} tokens` : ''}
          </div>
        </div>
      `
        }).join('')

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CodeMind Chat — ${projectName}</title>
  <style>
    body { background:#030712; margin:0; padding:24px; font-family:sans-serif; }
    h1   { color:#fff; font-size:20px; margin-bottom:8px; }
    p.meta { color:#6b7280; font-size:12px; margin-bottom:24px; }
  </style>
</head>
<body>
  <h1>🧠 CodeMind Chat Export</h1>
  <p class="meta">
    Project: <strong style="color:#a5b4fc">${projectName}</strong> ·
    Exported: ${new Date().toLocaleString()} ·
    ${messages.filter(m => m.role === 'user').length} questions
  </p>
  ${rows}
</body>
</html>`
}