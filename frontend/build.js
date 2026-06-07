const fs = require('fs');
const path = require('path');

const apiUrl = process.env.API_URL || '';
console.log(`[BUILD] Injecting API_URL: "${apiUrl}"`);

const htmlPath = path.join(__dirname, 'index.html');
if (fs.existsSync(htmlPath)) {
  let html = fs.readFileSync(htmlPath, 'utf8');
  html = html.replace('API_URL_PLACEHOLDER', apiUrl);
  fs.writeFileSync(htmlPath, html, 'utf8');
  console.log('[BUILD] Successfully injected API_URL into index.html');
} else {
  console.error('[BUILD] Error: index.html not found!');
  process.exit(1);
}
