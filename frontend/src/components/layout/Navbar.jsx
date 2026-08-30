// frontend/src/components/layout/Navbar.jsx
// COMPLETE REPLACEMENT

import { Link, useLocation, useParams } from 'react-router-dom'
import { Brain, Upload, MessageSquare, BookOpen } from 'lucide-react'

export default function Navbar() {
  const { pathname }  = useLocation()
  const { projectId } = useParams()

  const navItems = [
    {
      label: 'Upload',
      to:    '/',
      Icon:  Upload,
      active: pathname === '/',
    },
    {
      label: 'Chat',
      to:    projectId ? `/chat/${projectId}` : '/chat',
      Icon:  MessageSquare,
      active: pathname.startsWith('/chat'),
    },
    {
      label: 'Memory',
      to:    projectId ? `/memory/${projectId}` : '/memory',
      Icon:  BookOpen,
      active: pathname.startsWith('/memory'),
    },
  ]

  return (
    <nav className="fixed top-0 inset-x-0 z-50 h-14
                    bg-gray-950/90 backdrop-blur border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 h-full
                      flex items-center justify-between">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <Brain className="w-6 h-6 text-indigo-400
                            group-hover:text-indigo-300 transition-colors" />
          <span className="font-bold text-white tracking-tight">
            Code<span className="text-indigo-400">Mind</span>
          </span>
        </Link>

        {/* Navigation */}
        <div className="flex items-center gap-1">
          {navItems.map(({ label, to, Icon, active }) => (
            <Link
              key={label}
              to={to}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                          text-sm font-medium transition-colors
                          ${active
                            ? 'bg-indigo-600 text-white'
                            : 'text-gray-400 hover:text-white hover:bg-gray-800'
                          }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{label}</span>
            </Link>
          ))}
        </div>

        {/* API status */}
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <span className="hidden sm:inline">API</span>
        </div>
      </div>
    </nav>
  )
}