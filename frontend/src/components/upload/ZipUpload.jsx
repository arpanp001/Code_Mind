// frontend/src/components/upload/ZipUpload.jsx
import { useState, useRef, useCallback } from 'react'
import { Upload, FileArchive, X, Loader2 } from 'lucide-react'

export default function ZipUpload({ onSubmit, isLoading, progress }) {
  const [file,   setFile]   = useState(null)
  const [drag,   setDrag]   = useState(false)
  const [err,    setErr]    = useState('')
  const inputRef = useRef(null)

  const validate = f => {
    if (!f.name.endsWith('.zip'))           return 'Only .zip files accepted'
    if (f.size > 100 * 1024 * 1024)        return 'Max file size is 100MB'
    return ''
  }

  const pick = f => {
    const e = validate(f); if (e) { setErr(e); return }
    setErr(''); setFile(f)
  }

  const onDrop = useCallback(e => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]; if (f) pick(f)
  }, [])

  const fmtSize = b => b > 1e6
    ? `${(b/1e6).toFixed(1)} MB`
    : `${(b/1024).toFixed(0)} KB`

  return (
    <div className="card space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gray-800 flex items-center
                        justify-center">
          <FileArchive className="w-5 h-5 text-indigo-400" />
        </div>
        <div>
          <h3 className="font-semibold text-white text-sm">ZIP Upload</h3>
          <p className="text-xs text-gray-500">Max 100MB</p>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => !isLoading && inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center
                    cursor-pointer transition-all
                    ${drag   ? 'border-indigo-500 bg-indigo-500/10' :
                      file   ? 'border-green-500/40 bg-green-500/5' :
                               'border-gray-700 hover:border-gray-600 hover:bg-gray-800/40'}
                    ${isLoading ? 'cursor-not-allowed opacity-60' : ''}`}
      >
        {file ? (
          <div className="space-y-1">
            <FileArchive className="w-8 h-8 text-green-400 mx-auto" />
            <p className="text-white text-sm font-medium">{file.name}</p>
            <p className="text-gray-500 text-xs">{fmtSize(file.size)}</p>
            {!isLoading && (
              <button
                onClick={e => { e.stopPropagation(); setFile(null) }}
                className="text-xs text-gray-600 hover:text-red-400 mt-1
                           flex items-center gap-1 mx-auto transition-colors"
              >
                <X className="w-3 h-3" /> Remove
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="w-8 h-8 text-gray-600 mx-auto" />
            <p className="text-gray-400 text-sm">Drop ZIP here or click to browse</p>
          </div>
        )}
      </div>
      <input ref={inputRef} type="file" accept=".zip"
             onChange={e => e.target.files[0] && pick(e.target.files[0])}
             className="hidden" />

      {err && <p className="text-red-400 text-xs">{err}</p>}

      {/* Progress bar */}
      {isLoading && progress > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Uploading…</span><span>{progress}%</span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-500 rounded-full transition-all"
                 style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <button
        onClick={() => file && onSubmit(file)}
        disabled={!file || isLoading}
        className="btn-primary w-full flex items-center justify-center gap-2"
      >
        {isLoading
          ? <><Loader2 className="w-4 h-4 animate-spin" />
              {progress < 100 ? `Uploading ${progress}%…` : 'Processing…'}
            </>
          : <><Upload className="w-4 h-4" /> Upload & Analyze</>
        }
      </button>
    </div>
  )
}