// frontend/src/services/ingestService.js
// COMPLETE REPLACEMENT

import api from './api'
import axios from 'axios'

// Separate instance for preview — shorter timeout, doesn't kill the main request
const previewApi = axios.create({
    baseURL: '/api/v1',
    timeout: 10000,   // 10s — if GitHub API doesn't respond in 10s, show error
    headers: { 'Content-Type': 'application/json' },
})

previewApi.interceptors.response.use(
    r => r,
    error => {
        error.userMessage =
            error.code === 'ECONNABORTED'
                ? 'GitHub API took too long to respond. Check your connection or try again.'
                : error.response?.data?.detail || error.message || 'Request failed'
        return Promise.reject(error)
    }
)

export const ingestService = {

    // ── Ingestion ─────────────────────────────────────────────────────────────

    ingestGithub: async (url, branch = 'main', forceReindex = false) => {
        // Long timeout for actual cloning (large repos take minutes)
        const r = await api.post(
            `/ingest/github?force_reindex=${forceReindex}`,
            { url, branch },
            { timeout: 300000 }   // 5 minutes for large repos
        )
        return r.data
    },

    ingestZip: async (file, onProgress, forceReindex = false) => {
        const form = new FormData()
        form.append('file', file)
        const r = await api.post(
            `/ingest/zip?force_reindex=${forceReindex}`,
            form,
            {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000,   // 2 minutes for large ZIPs
                onUploadProgress: e => {
                    if (onProgress && e.total)
                        onProgress(Math.round((e.loaded * 100) / e.total))
                },
            }
        )
        return r.data
    },

    // ── GitHub helpers ────────────────────────────────────────────────────────

    previewGithubRepo: async (url) => {
        // Short timeout — this should be fast (metadata only, no clone)
        const r = await previewApi.get(
            `/ingest/github/preview?url=${encodeURIComponent(url)}`
        )
        return r.data
    },

    listBranches: async (url) => {
        const r = await previewApi.get(
            `/ingest/github/branches?url=${encodeURIComponent(url)}`
        )
        return r.data
    },

    // ── Project management ────────────────────────────────────────────────────

    getStatus: async (id) => (await api.get(`/ingest/status/${id}`)).data,
    listProjects: async () => (await api.get('/projects')).data,
    deleteProject: async (id) => (await api.delete(`/projects/${id}`)).data,
    getChromaStats: async (id) => (await api.get(`/projects/${id}/chroma-stats`)).data,
    reindexProject: async (projectId) => {
        const r = await api.post(`/projects/${projectId}/reindex`, {}, {
            timeout: 30000,   // Just starts the background task
        })
        return r.data
    },
    cancelIndexing: async (projectId) => {
        const r = await api.post(`/projects/${projectId}/cancel`)
        return r.data
    },
}