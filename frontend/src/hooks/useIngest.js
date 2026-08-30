// frontend/src/hooks/useIngest.js
//
// Fixes vs previous version:
//   1. 'exists' status correctly stores existingProject and does NOT
//      auto-navigate (lets HomePage show the Open/Re-index choice)
//   2. Polling stops immediately when status reaches terminal states
//      (ready, error, exists) — no more infinite polling loops
//   3. lastInput ref tracks what was submitted so Re-index can reuse it
//   4. reset() clears ALL state including existingProject
//   5. ingestZip/ingestGithub return the project_id for immediate use

import { useState, useCallback, useRef } from 'react'
import { ingestService } from '@/services/ingestService'

export function useIngest() {
  const [projectId,        setProjectId]        = useState(null)
  const [status,           setStatus]           = useState('idle')
  const [progress,         setProgress]         = useState(0)
  const [error,            setError]            = useState(null)
  const [projectInfo,      setProjectInfo]      = useState(null)
  const [existingProject,  setExistingProject]  = useState(null)

  const pollRef     = useRef(null)
  const lastInput   = useRef({ url: '', branch: 'main', file: null })

  // ── Stop polling ─────────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // ── Poll project status ───────────────────────────────────────────────────
  const startPolling = useCallback((pid) => {
    stopPolling()

    pollRef.current = setInterval(async () => {
      try {
        const data = await ingestService.getStatus(pid)
        setProjectInfo(data)

        if (data.status === 'ready') {
          setStatus('ready')
          stopPolling()
        } else if (data.status === 'failed') {
          setStatus('error')
          setError(data.error_message || 'Processing failed')
          stopPolling()
        }
        // Still processing — keep polling
      } catch (e) {
        setError(e.userMessage || 'Failed to check status')
        setStatus('error')
        stopPolling()
      }
    }, 3000)
  }, [stopPolling])

  // ── Ingest GitHub repo ────────────────────────────────────────────────────
  const ingestGithub = useCallback(async (url, branch = 'main', forceReindex = false) => {
    stopPolling()
    setError(null)
    setProgress(0)
    setExistingProject(null)
    setProjectInfo(null)
    setStatus('processing')
    lastInput.current = { url, branch, file: null }

    try {
      const result = await ingestService.ingestGithub(url, branch, forceReindex)

      // Backend returns existing project when already indexed
      const isExisting = (
        result.status === 'ready' ||
        (result.message && (
          result.message.includes('already indexed') ||
          result.message.includes('already exists')
        ))
      )

      if (isExisting && !forceReindex) {
        setExistingProject(result)
        setStatus('exists')
        setProjectId(result.project_id)
        return result.project_id
      }

      // New project — start polling for completion
      setProjectId(result.project_id)
      startPolling(result.project_id)
      return result.project_id

    } catch (e) {
      setStatus('error')
      setError(e.userMessage || e.message || 'GitHub ingestion failed')
      return null
    }
  }, [stopPolling, startPolling])

  // ── Ingest ZIP file ───────────────────────────────────────────────────────
  const ingestZip = useCallback(async (file, forceReindex = false) => {
    stopPolling()
    setError(null)
    setProgress(0)
    setExistingProject(null)
    setProjectInfo(null)
    setStatus('uploading')
    lastInput.current = { url: '', branch: 'main', file }

    try {
      const result = await ingestService.ingestZip(
        file,
        (pct) => {
          setProgress(pct)
          if (pct === 100) setStatus('processing')
        },
        forceReindex,
      )

      // Backend returns existing project when already indexed
      const isExisting = (
        result.status === 'ready' ||
        (result.message && (
          result.message.includes('already indexed') ||
          result.message.includes('already exists')
        ))
      )

      if (isExisting && !forceReindex) {
        setExistingProject(result)
        setStatus('exists')
        setProjectId(result.project_id)
        return result.project_id
      }

      // New project — poll for completion
      setProjectId(result.project_id)
      startPolling(result.project_id)
      return result.project_id

    } catch (e) {
      setStatus('error')
      setError(e.userMessage || e.message || 'ZIP upload failed')
      return null
    }
  }, [stopPolling, startPolling])

  // ── Force re-index ────────────────────────────────────────────────────────
  const forceReindex = useCallback(async () => {
    const { url, branch, file } = lastInput.current
    if (file) return ingestZip(file, true)
    if (url)  return ingestGithub(url, branch, true)
  }, [ingestGithub, ingestZip])

  // ── Reset all state ───────────────────────────────────────────────────────
  const reset = useCallback(() => {
    stopPolling()
    setProjectId(null)
    setStatus('idle')
    setProgress(0)
    setError(null)
    setProjectInfo(null)
    setExistingProject(null)
    lastInput.current = { url: '', branch: 'main', file: null }
  }, [stopPolling])

  return {
    projectId,
    status,
    progress,
    error,
    projectInfo,
    existingProject,
    lastInput,
    ingestGithub,
    ingestZip,
    forceReindex,
    reset,
    isLoading: ['uploading', 'processing'].includes(status),
  }
}