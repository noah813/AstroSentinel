document.addEventListener("DOMContentLoaded", () => {
    // Get the active tab and ask the content script for the bot count
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, { action: "getBotCount" }, (response) => {
                if (chrome.runtime.lastError) {
                    console.log("Content script not active on this page.");
                    return;
                }
                if (response && response.count !== undefined) {
                    document.getElementById("bot-count").innerText = response.count;
                }
            });
        }
    });
});
