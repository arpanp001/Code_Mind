// frontend/src/services/api.js
// COMPLETE REPLACEMENT

import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,   // 2 minutes — Gemini can take 30-60s on large contexts
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use(
  config => {
    if (import.meta.env.DEV) {
      console.log(`📤 ${config.method?.toUpperCase()} ${config.url}`)
    }
    return config
  },
  error => Promise.reject(error)
)

api.interceptors.response.use(
  response => response,
  error => {
    const isTimeout = (
      error.code === 'ECONNABORTED' ||
      error.message?.includes('timeout') ||
      error.message?.includes('60000ms') ||
      error.message?.includes('120000ms')
    )

    if (isTimeout) {
      error.userMessage = (
        'The AI is taking longer than expected. ' +
        'This often happens with large codebases. ' +
        'Please try a more specific question.'
      )
    } else {
      error.userMessage =
        error.response?.data?.detail ||
        error.response?.data?.error  ||
        error.message                ||
        'Something went wrong'
    }

    return Promise.reject(error)
  }
)

export default api