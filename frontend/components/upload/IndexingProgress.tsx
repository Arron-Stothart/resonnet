"use client"

import { useState, useEffect } from "react"
import { motion } from "motion/react"
import { Card, CardContent } from "@/components/ui/card"
import { claudeSearchApi, IndexingStatus } from "@/app/lib/api"
import { Trash, UploadIcon } from "lucide-react"

interface CheckmarkProps {
  size?: number
  strokeWidth?: number
  color?: string
  className?: string
}

const draw = {
  hidden: { pathLength: 0, opacity: 0 },
  visible: (i: number) => ({
    pathLength: 1,
    opacity: 1,
    transition: {
      pathLength: {
        delay: i * 0.2,
        type: "spring",
        duration: 1.5,
        bounce: 0.2,
        ease: "easeInOut",
      },
      opacity: { delay: i * 0.2, duration: 0.2 },
    },
  }),
}

export function Checkmark({ size = 100, strokeWidth = 2, color = "currentColor", className = "" }: CheckmarkProps) {
  return (
    <motion.svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      initial="hidden"
      animate="visible"
      className={className}
    >
      <title>Animated Checkmark</title>
      <motion.circle
        cx="50"
        cy="50"
        r="40"
        stroke={color}
        variants={draw}
        custom={0}
        style={{
          strokeWidth,
          strokeLinecap: "round",
          fill: "transparent",
        }}
      />
      <motion.path
        d="M30 50L45 65L70 35"
        stroke={color}
        variants={draw}
        custom={1}
        style={{
          strokeWidth,
          strokeLinecap: "round",
          strokeLinejoin: "round",
          fill: "transparent",
        }}
      />
    </motion.svg>
  )
}

interface IndexingProgressProps {
  initialStats: {
    has_conversations: boolean;
    conversation_count: number;
    message_count: number;
  } | null;
}

export default function IndexingProgress({ initialStats }: IndexingProgressProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [status, setStatus] = useState<IndexingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isComplete, setIsComplete] = useState(false)
  const [conversationStats, setConversationStats] = useState(initialStats)
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDeleteConversations = async () => {
    try {
      setIsDeleting(true)
      setError(null)
      await claudeSearchApi.deleteConversations()
      setConversationStats(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete conversations')
    } finally {
      setIsDeleting(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return
    
    try {
      setIsUploading(true)
      setError(null)
      
      const result = await claudeSearchApi.uploadFile(file)
      setTaskId(result.task_id)
      
      // Start polling for status updates
      pollStatus(result.task_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file')
    } finally {
      setIsUploading(false)
    }
  }

  const pollStatus = async (id: string) => {
    try {
      const status = await claudeSearchApi.getIndexingStatus(id)
      setStatus(status)
      
      // Continue polling if not completed or failed
      if (status.status === 'completed') {
        setIsComplete(true)
      } else if (status.status !== 'failed') {
        setTimeout(() => pollStatus(id), 2000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get indexing status')
    }
  }

  // File upload UI
  if (!taskId) {
    return (
      <Card className="w-full max-w-3xl mx-auto p-2 min-h-[300px] flex flex-col justify-start bg-[#262624] rounded-xl border-[0.1px] border-[#393936] backdrop-blur-sm">
        <CardContent className="space-y-6 pt-4">
          <div className="text-leading space-y-2">
            <h3 className="text-2xl text-[#E5E5E2] tracking-tighter font-display font-medium">
              Upload conversation history
            </h3>
          </div>
          
          {conversationStats?.has_conversations ? (
            <div className="p-4 bg-[#372324] rounded-2xl border-[0.1px] border-red-900 border-opacity-20">
              <p className="text-md text-[#EE8279] font-medium mb-4">
                You have {conversationStats.conversation_count} chat{conversationStats.conversation_count !== 1 ? 's' : ''} ({conversationStats.message_count} messages) indexed.
              </p>
              <div className="flex justify-center">
                <button
                  onClick={handleDeleteConversations}
                  className="flex flex-row items-center justify-center gap-2 px-4 py-2 bg-[#6F4038] text-[#EE8279] rounded-md hover:bg-[#5F3930] disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold text-md"
                >
                  <Trash className="w-4 h-4" />
                  {isDeleting ? 'Removing...' : 'Remove'}
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-[#262624] rounded-lg p-6 border border-[#393936] flex flex-col items-center justify-center gap-3 cursor-pointer">
              <label htmlFor="file-upload" className="w-full h-full flex flex-col items-center cursor-pointer">
              <div className="w-12 h-12 rounded-full bg-[#262624] flex items-center justify-center mb-3">
                <svg stroke="#C4C3BC" fill="none" stroke-width="1.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" className="w-8 h-8" xmlns="http://www.w3.org/2000/svg"><path d="M14 3v4a1 1 0 0 0 1 1h4"></path><path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z"></path><path d="M12 11v6"></path><path d="M9.5 13.5l2.5 -2.5l2.5 2.5"></path></svg>
              </div>
              <p className="text-[#C4C3BC] text-sm font-medium">
                {file ? 'Replace file' : 'Drag & drop or click to upload'}
              </p>
              {/* <p className="text-[#C4C3BC] text-xs mt-1">
                Supports conversation.json files
              </p> */}
              <input
                id="file-upload"
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="sr-only"
                disabled={isUploading}
              />
            </label>
          </div>
          )}

          
          {file && (
            <div className="flex items-center gap-2 p-3 bg-[#1A1918] rounded-lg">
              <svg 
                className="w-5 h-5 text-[#C4C3BC]" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="text-sm text-[#C4C3BC] truncate flex-1">
                {file.name}
              </span>
              <button 
                onClick={() => setFile(null)}
                className="text-[#C4C3BC] hover:text-[#E5E5E2] transition-colors"
              >
                <svg 
                  className="w-4 h-4" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
          
          <div className="flex flex-row gap-2 justify-end">
            <button
              onClick={() => setFile(null)}
              className="px-4 py-2 bg-[#232322] text-[#E5E5E2] rounded-md hover:bg-[#2F2F2E] disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold text-sm w-24"
            >
              Go Back
            </button>
            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              className="px-4 py-2 bg-[#AA542E] text-[#FFFFFF] rounded-md hover:bg-[#A1512B] disabled:bg-[#A1512B] disabled:cursor-not-allowed transition-colors font-semibold text-sm w-24"
            >
              {isUploading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Uploading...
                </span>
              ) : 'Confirm'}
            </button>
          </div>

          {error && (
            <div className="p-3 bg-[var(--error-bg)] border border-[var(--error-bg-secondary)] text-[var(--error-text)] rounded-lg text-sm">
              <div className="flex justify-start items-center gap-2 font-medium">
                <svg 
                  className="w-5 h-5 mt-0.5 flex-shrink-0" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{error}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }

  // Indexing progress UI
  if (status && !isComplete) {
    return (
      <Card className="w-full max-w-3xl mx-auto p-6 min-h-[300px] flex flex-col justify-start bg-[#262624] rounded-xl border-[0.1px] border-[#393936]  backdrop-blur-sm">
        <CardContent className="space-y-6 pt-4">
        <div className="text-leading space-y-2">
          <h3 className="text-2xl text-[#E5E5E2] tracking-tighter font-display font-medium">
            Processing conversation history
          </h3>
        </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-[#C4C3BC] mb-1">
              <span>{Math.round(status.progress * 100)}% complete</span>
              <span>{status.processed_messages} / {status.total_messages}</span>
            </div>
            <div className="w-full rounded-full h-2">
              <motion.div 
                className="bg-[#AA542E] h-2 rounded-full" 
                initial={{ width: 0 }}
                animate={{ width: `${Math.round(status.progress * 100)}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
          
          <div className="flex justify-center">
            <div className="animate-pulse flex space-x-2">
              <div className="w-2 h-2 bg-[var(--accent-main-100)] rounded-full"></div>
              <div className="w-2 h-2 bg-[var(--accent-main-100)] rounded-full animation-delay-200"></div>
              <div className="w-2 h-2 bg-[var(--accent-main-100)] rounded-full animation-delay-400"></div>
            </div>
          </div>
          
          {status.error && (
            <div className="p-3 bg-[var(--error-bg)] border border-[var(--error-bg-secondary)] text-[var(--error-text)] rounded-lg text-sm">
              <div className="flex items-start gap-2">
                <svg 
                  className="w-5 h-5 mt-0.5 flex-shrink-0" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{status.error}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full max-w-3xl mx-auto p-6 min-h-[300px] flex flex-col justify-center bg-[var(--bg-200)] dark:bg-[var(--bg-200)] border-[var(--border-200)] backdrop-blur-sm">
      <CardContent className="space-y-4 flex flex-col items-center justify-center">
        <motion.div
          className="flex justify-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: 0.4,
            ease: [0.4, 0, 0.2, 1],
            scale: {
              type: "spring",
              damping: 15,
              stiffness: 200,
            },
          }}
        >
          <div className="relative">
            <motion.div
              className="absolute inset-0 blur-xl bg-[var(--crail)]/10 dark:bg-[var(--crail)]/20 rounded-full"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                delay: 0.2,
                duration: 0.8,
                ease: "easeOut",
              }}
            />
            <Checkmark
              size={80}
              strokeWidth={4}
              color="var(--crail)"
              className="relative z-10 dark:drop-shadow-[0_0_10px_rgba(0,0,0,0.1)]"
            />
          </div>
        </motion.div>
        <motion.div
          className="space-y-2 text-center w-full"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            delay: 0.2,
            duration: 0.6,
            ease: [0.4, 0, 0.2, 1],
          }}
        >
          <motion.h2
            className="text-lg text-[var(--text-000)] tracking-tighter font-semibold uppercase"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.4 }}
          >
            Upload Successful
          </motion.h2>
          <div className="flex items-center gap-4">
            <motion.div
              className="flex-1 bg-[var(--bg-300)] dark:bg-[var(--bg-300)] rounded-xl p-3 border border-[var(--border-100)] backdrop-blur-md"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                delay: 1.2,
                duration: 0.4,
                ease: [0.4, 0, 0.2, 1],
              }}
            >
              <div className="flex flex-col items-start gap-2">
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-[var(--text-400)] flex items-center gap-1.5">
                    <svg
                      className="w-3 h-3"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <title>From</title>
                      <path d="M12 19V5M5 12l7-7 7 7" />
                    </svg>
                    From
                  </span>
                  <div className="flex items-center gap-2.5 group transition-all">
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--accent-main-100)] shadow-lg border border-[var(--accent-main-200)] text-sm font-medium text-[var(--pampas)] group-hover:scale-105 transition-transform">
                      $
                    </span>
                    <span className="font-medium text-[var(--text-100)] tracking-tight">500.00 USD</span>
                  </div>
                </div>
                <div className="w-full h-px bg-gradient-to-r from-transparent via-[var(--border-100)] to-transparent" />
                <div className="space-y-1.5">
                  <span className="text-xs font-medium text-[var(--text-400)] flex items-center gap-1.5">
                    <svg
                      className="w-3 h-3"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <title>To</title>
                      <path d="M12 5v14M5 12l7 7 7-7" />
                    </svg>
                    To
                  </span>
                  <div className="flex items-center gap-2.5 group transition-all">
                    <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-[var(--accent-main-100)] shadow-lg border border-[var(--accent-main-200)] text-sm font-medium text-[var(--pampas)] group-hover:scale-105 transition-transform">
                      €
                    </span>
                    <span className="font-medium text-[var(--text-100)] tracking-tight">460.00 EUR</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
          <motion.div
            className="w-full text-xs text-[var(--text-400)] mt-2 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.4, duration: 0.4 }}
          >
            Exchange Rate: 1 USD = 0.92 EUR
          </motion.div>
        </motion.div>
      </CardContent>
    </Card>
  )
}

