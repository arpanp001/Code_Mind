// frontend/src/context/ChatContext.jsx
//
// Global chat state store.
// Keeps message history alive even when navigating between pages.
// Each project gets its own message array and session ID.

import { createContext, useContext, useRef, useCallback, useState } from 'react'
import { queryService } from '@/services/queryService'

const ChatContext = createContext(null)

export function ChatProvider({ children }) {
  // Map of projectId → { messages, sessionId, isLoading, error }
  const chatStates    = useRef({})
  // Force re-renders when chat state changes
  const [tick, setTick] = useState(0)
  const rerender = () => setTick(t => t + 1)

  const getState = useCallback((projectId) => {
    if (!chatStates.current[projectId]) {
      chatStates.current[projectId] = {
        messages:  [],
        sessionId: crypto.randomUUID(),
        isLoading: false,
        error:     null,
      }
    }
    return chatStates.current[projectId]
  }, [])

  const sendMessage = useCallback(async (projectId, question) => {
    if (!projectId || !question.trim()) return

    const state = getState(projectId)
    if (state.isLoading) return

    // Add user message immediately
    state.messages = [...state.messages, {
      id:        Date.now(),
      role:      'user',
      content:   question,
      sources:   [],
      timestamp: new Date().toISOString(),
    }]
    state.isLoading = true
    state.error     = null
    rerender()

    try {
      const response = await queryService.chat(
        projectId,
        question,
        state.sessionId,
      )

      state.messages = [...state.messages, {
        id:           Date.now() + 1,
        role:         'assistant',
        content:      response.answer,
        sources:      response.sources || [],
        tokens:       response.tokens_used,
        memoriesUsed: response.memories_used || 0,
        timestamp:    new Date().toISOString(),
      }]
    } catch (e) {
      state.error = e.userMessage || 'Something went wrong'
      state.messages = [...state.messages, {
        id:        Date.now() + 1,
        role:      'error',
        content:   e.userMessage || 'Something went wrong. Please try again.',
        sources:   [],
        timestamp: new Date().toISOString(),
      }]
    } finally {
      state.isLoading = false
      rerender()
    }
  }, [getState])

  const clearChat = useCallback(async (projectId) => {
    const state = getState(projectId)
    state.messages = []
    state.error    = null
    // Clear backend session too
    try {
      await queryService.clearSession(state.sessionId)
    } catch (_) {}
    // Generate new session ID
    state.sessionId = crypto.randomUUID()
    rerender()
  }, [getState])

  const getProjectChat = useCallback((projectId) => {
    return getState(projectId)
  }, [getState])

  const getChatList = useCallback(() => {
  return Object.entries(chatStates.current)
    .filter(([, state]) => state.messages.length > 0)
    .map(([projectId, state]) => ({
      projectId,
      messageCount: state.messages.length,
      lastMessage: state.messages[state.messages.length - 1],
      sessionId: state.sessionId,
    }))
}, [])

  return (
  <ChatContext.Provider
    value={{
      sendMessage,
      clearChat,
      getProjectChat,
      getChatList,
    }}
  >
    {children}
  </ChatContext.Provider>
)
}

export function useChatContext(projectId) {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChatContext must be inside ChatProvider')

  const state = ctx.getProjectChat(projectId || '')
  return {
    messages:    state.messages,
    isLoading:   state.isLoading,
    error:       state.error,
    sessionId:   state.sessionId,
    sendMessage: (q) => ctx.sendMessage(projectId, q),
    clearChat:   ()  => ctx.clearChat(projectId),
  }
}