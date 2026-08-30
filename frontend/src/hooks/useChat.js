// frontend/src/hooks/useChat.js
import { useState, useCallback, useRef } from 'react'
import { queryService } from '@/services/queryService'

export function useChat(projectId) {
    const [messages, setMessages] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    // Persistent session ID for conversation memory
    const sessionId = useRef(crypto.randomUUID())

    const sendMessage = useCallback(async question => {
        if (!question.trim() || !projectId || isLoading) return

        const userMsg = {
            id: Date.now(),
            role: 'user',
            content: question,
            sources: [],
            timestamp: new Date().toISOString(),
        }
        setMessages(prev => [...prev, userMsg])
        setIsLoading(true)
        setError(null)

        try {
            const response = await queryService.chat(
                projectId,
                question,
                sessionId.current,
            )

            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                role: 'assistant',
                content: response.answer,
                sources: response.sources || [],
                tokens: response.tokens_used,
                memoriesUsed: response.memories_used || 0,   // ← ADD THIS
                timestamp: new Date().toISOString(),
            }])

        } catch (e) {
            setError(e.userMessage)
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                role: 'error',
                content: e.userMessage || 'Something went wrong. Please try again.',
                sources: [],
                timestamp: new Date().toISOString(),
            }])
        } finally {
            setIsLoading(false)
        }
    }, [projectId, isLoading])

    const clearChat = useCallback(async () => {
        setMessages([]); setError(null)
        try {
            await queryService.clearSession(sessionId.current)
        } catch (_) { }
        // Generate new session ID after clearing
        sessionId.current = crypto.randomUUID()
    }, [])

    return { messages, isLoading, error, sendMessage, clearChat, sessionId: sessionId.current }
}