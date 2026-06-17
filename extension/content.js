console.log("AstroSentinel Chrome Extension Loaded!");

const API_URL = "http://127.0.0.1:8000/check_bot";
const checkedCache = new Map();
let botCount = 0;
let isShieldActive = true;

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getBotCount") {
        sendResponse({ count: botCount });
    }
});

// Check if a user is a bot via background script (to bypass CORS/Mixed Content)
async function checkBotStatus(authorId) {
    if (!authorId) return false;
    if (checkedCache.has(authorId)) {
        return checkedCache.get(authorId);
    }
    return new Promise((resolve) => {
        chrome.runtime.sendMessage({ action: "checkBot", authorId: authorId }, (response) => {
            if (chrome.runtime.lastError || !response) {
                resolve(false);
            } else {
                checkedCache.set(authorId, response.isBot);
                resolve(response.isBot);
            }
        });
    });
}

// Toggle shield functionality
function toggleShield() {
    isShieldActive = !isShieldActive;
    const btn = document.getElementById("astrosentinel-toggle-btn");
    if (btn) {
        btn.innerText = isShieldActive ? "🛡️ 關閉 AstroSentinel 屏蔽" : "🛡️ 開啟 AstroSentinel 屏蔽";
        btn.style.backgroundColor = isShieldActive ? "#ff4a4a" : "#4a4a4a";
    }

    // Update all processed comments
    const botComments = document.querySelectorAll('.astrosentinel-bot-comment');
    for (let c of botComments) {
        const mainText = c.querySelector('#content-text') || c.querySelector('#expander');
        const badge = c.querySelector('.astrosentinel-badge');
        
        if (isShieldActive) {
            if (mainText) mainText.style.display = 'none';
            if (badge) badge.style.display = 'block';
            c.style.opacity = '0.5';
        } else {
            if (mainText) mainText.style.display = 'block';
            if (badge) badge.style.display = 'none';
            c.style.opacity = '1';
        }
    }
}

// Inject the toggle button into the YouTube comments header
function injectToggleButton() {
    if (document.getElementById("astrosentinel-toggle-btn")) return;
    
    // YouTube's comment header area
    const header = document.querySelector('ytd-comments-header-renderer #title');
    if (header) {
        const btn = document.createElement('button');
        btn.id = "astrosentinel-toggle-btn";
        btn.innerText = "🛡️ 關閉 AstroSentinel 屏蔽";
        btn.style.marginLeft = "20px";
        btn.style.padding = "6px 12px";
        btn.style.backgroundColor = "#ff4a4a";
        btn.style.color = "white";
        btn.style.border = "none";
        btn.style.borderRadius = "16px";
        btn.style.cursor = "pointer";
        btn.style.fontWeight = "bold";
        btn.style.fontSize = "13px";
        btn.onclick = toggleShield;
        
        header.appendChild(btn);
    }
}

// Function to scan and process YouTube comments
async function processComments() {
    injectToggleButton();
    
    const comments = document.querySelectorAll('ytd-comment-view-model:not(.astrosentinel-processed), ytd-comment-renderer:not(.astrosentinel-processed)');
    
    for (let comment of comments) {
        comment.classList.add('astrosentinel-processed');
        
        // Find the author link. It can be #author-text or any link containing /@ or /channel/
        let authorLink = comment.querySelector('#author-text');
        if (!authorLink) {
            // Fallback for newer YouTube layouts
            const links = comment.querySelectorAll('a[href^="/@"], a[href^="/channel/"]');
            if (links.length > 0) authorLink = links[0];
        }
        if (!authorLink) continue;
        
        let authorId = authorLink.getAttribute('href'); 
        if (authorId) {
            authorId = authorId.split('?')[0].replace('/', ''); 
        }
        
        const isBot = await checkBotStatus(authorId);
        
        if (isBot) {
            botCount++;
            comment.classList.add('astrosentinel-bot-comment');
            
            const mainText = comment.querySelector('#content-text') || comment.querySelector('#expander');
            
            // Create warning badge
            const warningBadge = document.createElement('div');
            warningBadge.className = 'astrosentinel-badge';
            warningBadge.style.color = '#ff4a4a';
            warningBadge.style.fontSize = '13px';
            warningBadge.style.fontWeight = 'bold';
            warningBadge.style.padding = '8px 0';
            warningBadge.innerText = '🛡️ 此留言已被 AstroSentinel 自動屏蔽 (已知水軍)';
            
            const mainBody = comment.querySelector('#main') || comment;
            mainBody.insertBefore(warningBadge, mainBody.querySelector('#expander') || mainBody.firstChild);
            
            if (isShieldActive) {
                if (mainText) mainText.style.display = 'none';
                comment.style.opacity = '0.5';
            } else {
                warningBadge.style.display = 'none';
            }
        }
    }
}

const observer = new MutationObserver((mutations) => {
    let shouldProcess = false;
    for (let mutation of mutations) {
        if (mutation.addedNodes.length > 0) {
            shouldProcess = true;
            break;
        }
    }
    if (shouldProcess) {
        processComments();
    }
});

observer.observe(document.body, { childList: true, subtree: true });
setTimeout(processComments, 2000);
