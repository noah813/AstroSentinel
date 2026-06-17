chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "checkBot") {
        const url = `http://127.0.0.1:8000/check_bot/${encodeURIComponent(request.authorId)}`;
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error("Network response was not ok");
                return res.json();
            })
            .then(data => sendResponse({ isBot: data.is_bot }))
            .catch(error => sendResponse({ isBot: false }));
        
        // Return true to indicate asynchronous response
        return true; 
    }
});
