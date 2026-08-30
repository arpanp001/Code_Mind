import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

export function useKeyboardShortcuts(projectId) {
    const navigate = useNavigate()

    useEffect(() => {
        const handler = (e) => {
            const tag = document.activeElement?.tagName

            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return

            if (e.ctrlKey || e.metaKey) {
                switch (e.key.toLowerCase()) {
                    case 'u':
                        e.preventDefault()
                        navigate('/')
                        break

                    case 'm':
                        e.preventDefault()
                        navigate(projectId ? `/memory/${projectId}` : '/memory')
                        break

                    case 'k':
                        e.preventDefault()
                        navigate(projectId ? `/chat/${projectId}` : '/chat')
                        break

                    default:
                        break
                }
            }
        }

        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [navigate, projectId])
}