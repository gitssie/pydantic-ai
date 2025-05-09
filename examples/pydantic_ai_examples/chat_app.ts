// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking

import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
const convElement = document.getElementById('conversation')

const promptInput = document.getElementById('prompt-input') as HTMLTextAreaElement
const spinner = document.getElementById('spinner')

// 存储已处理的消息ID
const processedMessageIds = new Set<string>()

// 配置marked渲染器以支持emoji和代码高亮
marked.use({
  gfm: true,
  breaks: true
})

// 自动调整文本区域高度
function autoResizeTextarea() {
  promptInput.style.height = 'auto'
  promptInput.style.height = (Math.min(promptInput.scrollHeight, 200)) + 'px'
}

// 在文本区域输入时自动调整高度
promptInput.addEventListener('input', autoResizeTextarea)

// 在按下Enter键但没有按Shift键时提交表单
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    const form = promptInput.closest('form')
    if (form) {
      const event = new Event('submit', { cancelable: true })
      form.dispatchEvent(event)
    }
  }
})

// 流式处理响应并在收到每个数据块时渲染消息
// 数据以换行符分隔的JSON发送
async function onFetchResponse(response: Response): Promise<void> {
  let text = ''
  let decoder = new TextDecoder()
  if (response.ok) {
    const reader = response.body!.getReader()
    while (true) {
      const {done, value} = await reader.read()
      if (done) {
        break
      }
      text += decoder.decode(value)
      try {
        renderMessages(text)
        spinner!.classList.remove('active')
      } catch (e) {
        console.warn('Error parsing streamed response', e)
      }
    }
    renderMessages(text)
    promptInput.disabled = false
    promptInput.value = ''
    promptInput.style.height = 'auto'
    promptInput.focus()
  } else {
    const errorText = await response.text()
    console.error(`HTTP错误: ${response.status}`, {response, errorText})
    document.getElementById('error')!.classList.remove('d-none')
    document.getElementById('error')!.textContent = `请求失败 (${response.status}): ${errorText || '未知错误'}`
    spinner!.classList.remove('active')
    promptInput.disabled = false
    throw new Error(`请求失败: ${response.status}`)
  }
}

// 消息的格式，这与pydantic-ai匹配
interface Message {
  role: string
  content: string
  timestamp: string
  uid: string  // 添加uid字段
}

// 处理原始响应文本并将消息渲染到`#conversation`元素中
function renderMessages(responseText: string) {
  if (!responseText.trim()) return
  
  // 分割响应并解析每一行JSON
  const lines = responseText.split('\n')
  const messages: Message[] = []
  
  // 安全地尝试解析每一行
  for (const line of lines) {
    if (line.trim().length <= 1) continue
    try {
      const msg = JSON.parse(line)
      messages.push(msg)
    } catch (e) {
      console.warn(`无法解析消息行: ${line}`, e)
    }
  }
  
  if (messages.length === 0) return

  // 处理所有有效消息
  for (const message of messages) {
    const {timestamp, role, content, uid} = message
    
    // 创建消息ID
    const messageId = `msg-${uid}`
    
    // 如果消息已经处理过，跳过
    if (processedMessageIds.has(messageId)) continue
    
    // 创建消息组元素
    const groupElement = document.createElement('div')
    groupElement.id = messageId
    groupElement.className = `message-group ${role === 'user' ? 'user-group' : 'ai-group'}`
    
    // 添加发送者标签
    const labelElement = document.createElement('div')
    labelElement.className = 'message-label'
    labelElement.textContent = role === 'user' ? 'You' : 'AI Assistant'
    groupElement.appendChild(labelElement)
    
    // 创建消息气泡元素
    const bubbleElement = document.createElement('div')
    bubbleElement.className = `message-bubble ${role === 'user' ? 'user-message' : 'ai-message'}`
    
    // 设置气泡内容
    try {
      // 规范化emoji在工具调用提示中的显示
      let processedContent = content
        .replace(/⚙️ 调用工具:/g, "⚙️ *调用工具:*")
        .replace(/📊 工具结果:/g, "📊 *工具结果:*")
      
      bubbleElement.innerHTML = marked.parse(processedContent)
    } catch (e) {
      console.warn('Markdown解析错误', e)
      bubbleElement.textContent = content
    }
    
    // 添加气泡到消息组
    groupElement.appendChild(bubbleElement)
    
    // 添加消息组到对话容器
    convElement!.appendChild(groupElement)
    
    // 记录已处理的消息ID
    processedMessageIds.add(messageId)
  }
  
  // 滚动到底部
  convElement!.scrollTop = convElement!.scrollHeight
}

// 错误处理
function onError(error: any) {
  console.error('操作失败', error)
  const errorEl = document.getElementById('error')!
  errorEl.classList.remove('d-none')
  if (error && error.message) {
    errorEl.textContent = `错误: ${error.message}`
  }
  spinner!.classList.remove('active')
  promptInput.disabled = false
}

// 表单提交处理
async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  
  // 获取输入文本
  const inputText = promptInput.value.trim()
  if (!inputText) return
  
  // 显示加载指示器
  spinner!.classList.add('active')
  
  // 禁用输入区域
  promptInput.disabled = true
  
  // 准备请求数据
  const body = new FormData(e.target as HTMLFormElement)
  
  try {
    // 发送请求
    const response = await fetch('/chat/', {method: 'POST', body})
    await onFetchResponse(response)
  } catch (error) {
    onError(error)
  }
}

// 监听表单提交事件
document.querySelector('form')!.addEventListener('submit', (e) => onSubmit(e).catch(onError))

// 页面加载时获取消息历史
fetch('/chat/').then(onFetchResponse).catch(onError)
