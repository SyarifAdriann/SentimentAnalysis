const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');
const puppeteer = require('puppeteer');
const dotenv = require('dotenv');

dotenv.config();

const SERVER_URL = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const TIMEOUT = 20000;

const ADMIN_USERNAME = process.env.ADMIN_USERNAME || 'admin';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'supervisor';

const headed = process.argv.includes('--headed');

function resolvePythonExecutable() {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }
  if (process.platform === 'win32') {
    return path.join(process.cwd(), 'venv', 'Scripts', 'python.exe');
  }
  return path.join(process.cwd(), 'venv', 'bin', 'python');
}

function startServer() {
  const pythonExec = resolvePythonExecutable();
  const server = spawn(pythonExec, ['run.py'], {
    cwd: process.cwd(),
    env: { ...process.env, FLASK_ENV: 'development' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return { server, ready: waitForServerReady(server) };
}

function waitForServerReady(server) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Timed out waiting for Flask server to start'));
    }, 20000);

    const handleData = (data) => {
      const text = data.toString();
      if (text.includes('Running on http://') || text.includes('Running on https://')) {
        clearTimeout(timeout);
        server.stdout.off('data', handleData);
        server.stderr.off('data', stderrHandler);
        resolve();
      }
    };

    const stderrHandler = (data) => {
      process.stderr.write(data);
      handleData(data);
    };

    server.stdout.on('data', handleData);
    server.stderr.on('data', stderrHandler);
    server.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

async function ensureAdminCredentials(page, { username, password }) {
  await page.waitForSelector('form', { timeout: TIMEOUT });
  await page.evaluate(({ username, password }) => {
    const form = document.querySelector('form');
    if (!form) {
      throw new Error('Login form not rendered');
    }
    let usernameInput = form.querySelector('input[name="username"]');
    if (!usernameInput) {
      usernameInput = document.createElement('input');
      usernameInput.name = 'username';
      usernameInput.type = 'text';
      usernameInput.value = '';
      usernameInput.style.display = 'none';
      form.insertBefore(usernameInput, form.firstChild);
    }
    usernameInput.value = username;

    const passwordInput = form.querySelector('input[name="password"]');
    if (!passwordInput) {
      throw new Error('Password field missing on admin login form');
    }
    passwordInput.value = password;
  }, { username, password });
}

async function runTests() {
  const { server, ready } = startServer();
  let browser;
  const results = [];
  const context = {};

  try {
    await ready;
    browser = await puppeteer.launch({ headless: !headed });
    const page = await browser.newPage();
    page.setDefaultTimeout(TIMEOUT);

    const tests = [
      {
        name: 'loads home page',
        run: async () => {
          await page.goto(`${SERVER_URL}/`, { waitUntil: 'networkidle2' });
          const title = await page.title();
          if (!title.includes('Predict Sentiment')) {
            throw new Error(`Unexpected title: ${title}`);
          }
        },
      },
      {
        name: 'submits sentiment prediction',
        run: async () => {
          const sampleText = `Puppeteer test flight ${Date.now()}`;
          context.latestTweet = sampleText;
          await page.type('textarea[name="tweet_text"]', sampleText);
          await Promise.all([
            page.click('input[type="submit"]'),
            page.waitForNavigation({ waitUntil: 'networkidle0' }),
          ]);
          await page.waitForSelector('.flash-success', { timeout: 5000 });
        },
      },
      {
        name: 'recent submissions shows latest entry',
        run: async () => {
          const selector = '.data-table tbody tr:first-child td:first-child';
          await page.waitForSelector(selector, { timeout: 5000 });
          const textSnippet = await page.$eval(selector, (el) => el.textContent || '');
          if (!textSnippet.includes(context.latestTweet.split(' ')[0])) {
            throw new Error('Latest submission not found in recent table');
          }
        },
      },
      {
        name: 'dashboard displays key visuals',
        run: async () => {
          await page.goto(`${SERVER_URL}/dashboard`, { waitUntil: 'networkidle2' });
          const viz = await page.$('img[alt="Sentiment distribution chart"]');
          if (!viz) {
            throw new Error('Sentiment visualization not found');
          }
          const iframe = await page.$('iframe');
          if (!iframe) {
            throw new Error('Interactive iframe missing');
          }
        },
      },
      {
        name: 'admin login rejects wrong password',
        run: async () => {
          await page.goto(`${SERVER_URL}/admin`, { waitUntil: 'networkidle2' });
          await ensureAdminCredentials(page, { username: ADMIN_USERNAME, password: 'wrongpass' });
          await Promise.all([
            page.click('input[type="submit"]'),
            page.waitForNavigation({ waitUntil: 'networkidle0' }),
          ]);
          const error = await page.$('.flash-danger');
          if (!error) {
            throw new Error('Expected danger flash message');
          }
        },
      },
      {
        name: 'admin login succeeds with valid password',
        run: async () => {
          await ensureAdminCredentials(page, { username: ADMIN_USERNAME, password: ADMIN_PASSWORD });
          await Promise.all([
            page.click('input[type="submit"]'),
            page.waitForNavigation({ waitUntil: 'networkidle0' }),
          ]);
          const postLogin = await page.content();
          fs.writeFileSync(path.join(process.cwd(), 'logs', 'debug-admin-success.html'), postLogin, 'utf8');
          await page.waitForSelector('.flash-success', { timeout: 5000 });
          context.isAdmin = true;
        },
      },
      {
        name: 'admin can approve pending submission',
        run: async () => {
          if (!context.isAdmin) {
            throw new Error('Admin session not established');
          }
          const formSelector = '.inline-form';
          const form = await page.$(formSelector);
          if (!form) {
            return;
          }
          await page.select(`${formSelector} select[name="true_sentiment"]`, 'positive');
          await page.type(`${formSelector} input[name="admin_comment"]`, 'Automated approval');
          await Promise.all([
            page.click(`${formSelector} button[value="approve"]`),
            page.waitForNavigation({ waitUntil: 'networkidle0' }),
          ]);
          const success = await page.$('.flash-success');
          if (!success) {
            throw new Error('Approval confirmation not displayed');
          }
        },
      },
    ];

    for (const test of tests) {
      try {
        await test.run();
        results.push({ name: test.name, status: 'passed' });
        console.log(`? ${test.name}`);
      } catch (error) {
        results.push({ name: test.name, status: 'failed', error: error.message });
        console.error(`? ${test.name} -> ${error.message}`);
      }
    }
  } catch (err) {
    console.error('Test harness error:', err);
    results.push({ name: 'harness', status: 'failed', error: err.message });
  } finally {
    if (browser) {
      await browser.close();
    }
  }

  if (server && !server.killed) {
    server.kill('SIGTERM');
  }

  const failed = results.filter((item) => item.status !== 'passed');
  console.log('\nTest Summary:');
  results.forEach((item) => {
    console.log(` - ${item.status === 'passed' ? 'PASS' : 'FAIL'} :: ${item.name}${item.error ? ' -> ' + item.error : ''}`);
  });

  const resultsPath = path.join(process.cwd(), 'logs', 'puppeteer_results.json');
  fs.mkdirSync(path.dirname(resultsPath), { recursive: true });
  fs.writeFileSync(
    resultsPath,
    JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2),
    'utf8'
  );

  if (failed.length > 0) {
    process.exitCode = 1;
  }
}

runTests();
