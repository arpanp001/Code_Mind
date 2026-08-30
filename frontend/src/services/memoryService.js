// frontend/src/services/memoryService.js
// COMPLETE REPLACEMENT

import api from './api'

export const memoryService = {

    addMemory: async (projectId, content, memoryType, tags = [], title = '') => {
        const r = await api.post('/memory/add', {
            project_id: projectId,
            content,
            memory_type: memoryType,
            tags,
            title,
        })
        return r.data
    },

    listMemories: async (projectId) => {
        const r = await api.get(`/memory/${projectId}`)
        return r.data
    },

    searchMemories: async (projectId, query) => {
        const r = await api.post('/memory/search', {
            project_id: projectId,
            query,
            top_k: 10,
        })
        return r.data
    },

    // project_id sent as query param for the DELETE endpoint
    deleteMemory: async (memoryId, projectId) => {
        const r = await api.delete(
            `/memory/${memoryId}`,
            { params: { project_id: projectId } }
        )
        return r.data
    },
}