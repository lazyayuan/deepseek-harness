import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { once } from 'node:events'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import McpResources from '@deepseek-ai/dsh-mcp-resources'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import { executeToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import { Config } from '@deepseek-ai/dsh-mcp-client'
import { startConnection, resolveReconnectPolicy } from '@deepseek-ai/dsh-mcp-client/src/connection.ts'
import { McpStatusCell } from '@deepseek-ai/dsh-mcp-client/src/status.ts'
import { describe, expect, it, onTestFinished, vi } from 'vitest'

/** A legacy stateful server whose sessions can expire without closing the client transport. */
async function fixture(stateful = true) {
  let session: string | undefined
  let initialized = 0
  let calls = 0
  let rejectionStatus: number | undefined
  const handle = async (request: IncomingMessage, response: ServerResponse): Promise<void> => {
    if (request.method !== 'POST') {
      response.writeHead(405).end()
      return
    }
    request.setEncoding('utf8')
    let body = ''
    for await (const chunk of request) {
      if (typeof chunk !== 'string') throw new Error('Expected UTF-8 request body')
      body += chunk
    }
    const message = JSON.parse(body) as { id?: number; method: string }
    const reply = (status: number, body: object) => {
      response.writeHead(status, { 'content-type': 'application/json' }).end(JSON.stringify(body))
    }
    if (message.method === 'initialize') {
      initialized++
      session = stateful ? `session-${initialized}` : undefined
      if (session !== undefined) response.setHeader('mcp-session-id', session)
      reply(200, { jsonrpc: '2.0', id: message.id, result: {
        protocolVersion: '2025-06-18', capabilities: { tools: {}, resources: {} },
        serverInfo: { name: 'expiry-fixture', version: '1.0.0' },
      } })
      return
    }
    if (message.method === 'server/discover') {
      reply(200, { jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Method not found' } })
      return
    }
    if (message.method.startsWith('notifications/')) {
      response.writeHead(202).end()
      return
    }
    if (rejectionStatus !== undefined || stateful && request.headers['mcp-session-id'] !== session) {
      reply(rejectionStatus ?? 404, { code: -32002, message: 'Session not found' })
      return
    }
    if (message.method === 'tools/list') {
      reply(200, { jsonrpc: '2.0', id: message.id, result: {
        tools: [{ name: 'ping', description: 'Replies pong.', inputSchema: { type: 'object' } }],
      } })
      return
    }
    if (message.method === 'resources/list') {
      reply(200, { jsonrpc: '2.0', id: message.id, result: { resources: [] } })
      return
    }
    calls++
    reply(200, { jsonrpc: '2.0', id: message.id, result: { content: [{ type: 'text', text: 'pong' }] } })
  }
  const server = createServer((request, response) => {
    void handle(request, response).catch((error: unknown) => { response.writeHead(500).end(String(error)) })
  })
  onTestFinished(async () => {
    server.closeAllConnections()
    await new Promise<void>((resolve, rejectClose) => {
      server.close((error) => { if (error) rejectClose(error); else resolve() })
    })
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('Missing fixture address')
  return {
    url: `http://127.0.0.1:${address.port}/mcp`,
    expire: () => { session = undefined },
    reject: (status = 404) => { rejectionStatus = status },
    initialized: () => initialized,
    calls: () => calls,
  }
}

async function mount(url: string, enabled = true) {
  const ctx = new Context()
  onTestFinished(() => ctx.fiber.dispose())
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(NativeToolExecutor)
  await ctx.plugin(McpResources)
  const config = Config({ transport: 'streamable-http', serverName: 'expiry', url,
    startupTimeoutMs: 5000, closeTimeoutMs: 1000,
    reconnect: { enabled, initialDelayMs: 5, maxDelayMs: 20, maxAttempts: 2 },
  })
  const status = new McpStatusCell(() => {})
  const handle = startConnection(ctx, config, resolveReconnectPolicy(config.reconnect, 'test'), status)
  onTestFinished(() => handle.dispose())
  expect(await handle.ready).toEqual({})
  const unregister = ctx.mcpResources.register('expiry', handle.resources)
  onTestFinished(unregister)
  let callID = 0
  return {
    status,
    dispose: () => handle.dispose(),
    resources: () => executeToolFixture(ctx, { name: 'list_mcp_resources', arguments: { server: 'expiry' },
      callId: ToolCallId(`expiry-${++callID}`), signal: new AbortController().signal,
    }),
    call: () => executeToolFixture(ctx, { name: 'mcp__expiry__ping', arguments: {},
      callId: ToolCallId(`expiry-${++callID}`), signal: new AbortController().signal,
    }),
  }
}

describe('expired Streamable HTTP sessions', () => {
  it('reinitializes after a session 404 without replaying the failed tool call', async () => {
    const server = await fixture()
    const client = await mount(server.url)
    expect((await client.call()).isError).toBe(false)
    server.expire()
    expect((await client.call()).isError).toBe(true)
    await vi.waitFor(() => {
      expect(server.initialized()).toBe(2)
      expect(client.status.current().state).toBe('connected')
    })
    expect(server.calls()).toBe(1)
    expect((await client.call()).isError).toBe(false)
    expect(server.calls()).toBe(2)
  })

  it('recovers session expiry observed by a resource request', async () => {
    const server = await fixture()
    const client = await mount(server.url)
    server.expire()
    const result = await client.resources()
    expect(result.isError).toBe(true)
    if (result.isError) expect(result.error.message).toContain('MCP')
    await vi.waitFor(() => {
      expect(server.initialized()).toBe(2)
      expect(client.status.current()).toMatchObject({ state: 'connected' })
    })
    expect(server.initialized()).toBe(2)
    expect((await client.resources()).isError).toBe(false)
  })

  it('coalesces concurrent expiry failures into one replacement', async () => {
    const server = await fixture()
    const client = await mount(server.url)
    server.expire()
    const results = await Promise.all([client.call(), client.call()])
    expect(results.every(result => result.isError)).toBe(true)
    await vi.waitFor(() => {
      expect(server.initialized()).toBe(2)
      expect(client.status.current().state).toBe('connected')
    })
    expect(server.calls()).toBe(0)
  })

  it('stops at the retry budget when the endpoint remains unavailable', async () => {
    const server = await fixture()
    const client = await mount(server.url)
    server.reject()
    expect((await client.call()).isError).toBe(true)
    await vi.waitFor(() => { expect(client.status.current()).toMatchObject({ state: 'failed', reason: 'gave-up' }) })
    expect(server.initialized()).toBe(3)
    expect(server.calls()).toBe(0)
  })

  it('does not replace a disposed connection after expiry', async () => {
    const server = await fixture()
    const client = await mount(server.url)
    let disposing: Promise<void> | undefined
    client.status.onChange((value) => { if (value.state === 'reconnecting') disposing = client.dispose() })
    server.expire()
    expect((await client.call()).isError).toBe(true)
    await vi.waitFor(() => { expect(disposing).toBeDefined() })
    await disposing
    expect(client.status.current().state).toBe('closed')
    expect(server.initialized()).toBe(1)
  })

  it('does not recover a 404 on a connection without a session ID', async () => {
    const server = await fixture(false)
    const client = await mount(server.url)
    server.reject()
    expect((await client.call()).isError).toBe(true)
    expect(client.status.current().state).toBe('connected')
    expect(server.initialized()).toBe(1)
  })

  it.each([400, 401, 500])('does not classify HTTP %s as session expiry', async (status) => {
    const server = await fixture()
    const client = await mount(server.url)
    server.reject(status)
    expect((await client.call()).isError).toBe(true)
    expect(client.status.current().state).toBe('connected')
    expect(server.initialized()).toBe(1)
  })

  it('honors disabled reconnection after session expiry', async () => {
    const server = await fixture()
    const client = await mount(server.url, false)
    server.expire()
    expect((await client.call()).isError).toBe(true)
    await vi.waitFor(() => { expect(client.status.current()).toMatchObject({ state: 'failed', reason: 'reconnect-disabled' }) })
    expect(server.initialized()).toBe(1)
  })
})
