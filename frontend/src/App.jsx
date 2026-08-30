import {
  BrowserRouter,
  Routes,
  Route,
  useLocation,
} from 'react-router-dom'

import { ChatProvider } from '@/context/ChatContext'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'

import Navbar     from '@/components/layout/Navbar'
import HomePage   from '@/pages/HomePage'
import ChatPage   from '@/pages/ChatPage'
import MemoryPage from '@/pages/MemoryPage'


function AppShell({ children }) {
  const location = useLocation()

  const match = location.pathname.match(
    /\/(?:chat|memory)\/([^/]+)/
  )

  const projectId = match?.[1]

  useKeyboardShortcuts(projectId)

  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <ChatProvider>
        <AppShell>
          <Navbar />

          <Routes>
            <Route path="/"                  element={<HomePage />} />
            <Route path="/chat"              element={<ChatPage />} />
            <Route path="/chat/:projectId"   element={<ChatPage />} />
            <Route path="/memory"            element={<MemoryPage />} />
            <Route path="/memory/:projectId" element={<MemoryPage />} />

            <Route
              path="*"
              element={
                <div className="min-h-screen flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <p className="text-5xl">🤔</p>
                    <h1 className="text-white text-2xl font-bold">
                      Page not found
                    </h1>
                    <a href="/" className="btn-primary inline-block">
                      Go Home
                    </a>
                  </div>
                </div>
              }
            />
          </Routes>

        </AppShell>
      </ChatProvider>
    </BrowserRouter>
  )
}