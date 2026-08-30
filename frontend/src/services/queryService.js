// frontend/src/services/queryService.js
import api from './api'

export const queryService = {
    chat: async (projectId, question, sessionId, options = {}) => {
        const r = await api.post('/query/chat', {
            project_id: projectId,
            question,
            session_id: sessionId,
            include_sources: options.includeSources ?? true,
            max_sources: options.maxSources ?? 5,
        })
        return r.data
    },

    search: async (projectId, query, topK = 5) => {
        const r = await api.post('/query/search', {
            project_id: projectId,
            query,
            top_k: topK,
        })
        return r.data
    },

    explain: async (projectId, code, language, filePath = '', question = '') => {
        const r = await api.post('/query/explain', {
            project_id: projectId,
            code,
            language,
            file_path: filePath,
            question,
        })
        return r.data
    },

    clearSession: async sessionId => {
        const r = await api.delete(`/query/session/${sessionId}`)
        return r.data
    },
}