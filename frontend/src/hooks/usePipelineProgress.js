// frontend/src/hooks/usePipelineProgress.js

import { useState, useEffect, useRef } from 'react'

const STEP_LABELS = {
    cloning: { label: 'Cloning repository', icon: '📥', color: 'text-blue-400' },
    extracting: { label: 'Extracting ZIP archive', icon: '📦', color: 'text-blue-400' },
    parsing: { label: 'Parsing source files', icon: '🔍', color: 'text-purple-400' },
    chunking: { label: 'Chunking code', icon: '✂️', color: 'text-indigo-400' },
    embedding: { label: 'Generating embeddings', icon: '🧮', color: 'text-cyan-400' },
    storing: { label: 'Storing in ChromaDB', icon: '💾', color: 'text-green-400' },
    summarizing: { label: 'Generating summary', icon: '📋', color: 'text-yellow-400' },
    done: { label: 'Indexing complete!', icon: '✅', color: 'text-green-400' },
    error: { label: 'Error occurred', icon: '❌', color: 'text-red-400' },
}

export function usePipelineProgress(projectId, isActive) {
    const [progress, setProgress] = useState(null)
    const esRef = useRef(null)

    useEffect(() => {
        if (!projectId || !isActive) {
            setProgress(null)
            return
        }

        // Connect to SSE endpoint
        const es = new EventSource(`/api/v1/ingest/progress/${projectId}`)
        esRef.current = es

        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data)
                setProgress(data)

                // Auto-close when done or error
                if (data.step === 'done' || data.step === 'error') {
                    es.close()
                }
            } catch (_) { }
        }

        es.onerror = () => {
            // SSE connection closed — this is normal when pipeline finishes
            es.close()
        }

        return () => {
            es.close()
        }
    }, [projectId, isActive])

    const stepInfo = progress ? (STEP_LABELS[progress.step] || STEP_LABELS.cloning) : null

    return { progress, stepInfo }
}