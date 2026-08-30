// frontend/src/utils/formatters.js

export function timeAgo(isoString) {
    if (!isoString) return ''
    const diff = Date.now() - new Date(isoString).getTime()
    const mins = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)
    if (mins < 1) return 'just now'
    if (mins < 60) return `${mins}m ago`
    if (hours < 24) return `${hours}h ago`
    return `${days}d ago`
}

export function formatBytes(bytes) {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}

export function getLanguageColor(lang) {
    const colors = {
        python: 'text-blue-400',
        javascript: 'text-yellow-400',
        typescript: 'text-blue-300',
        java: 'text-orange-400',
        cpp: 'text-purple-400',
        csharp: 'text-green-400',
        go: 'text-cyan-400',
        rust: 'text-orange-300',
        markdown: 'text-gray-400',
        default: 'text-gray-400',
    }
    return colors[lang?.toLowerCase()] || colors.default
}

export function truncate(str, n = 60) {
    return str?.length > n ? str.slice(0, n) + '…' : str
}

export function getMemoryTypeInfo(type) {
    const map = {
        architecture_decision: { label: 'Architecture', emoji: '🏗️', color: 'text-blue-400   bg-blue-400/10  border-blue-400/20' },
        bug_fix: { label: 'Bug Fix', emoji: '🐛', color: 'text-red-400    bg-red-400/10   border-red-400/20' },
        note: { label: 'Note', emoji: '📝', color: 'text-green-400  bg-green-400/10 border-green-400/20' },
    }
    return map[type] || map.note
}