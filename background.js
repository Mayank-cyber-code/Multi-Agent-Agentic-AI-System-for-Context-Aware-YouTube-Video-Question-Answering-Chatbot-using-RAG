// background.js

chrome.runtime.onInstalled.addListener(() => {
  console.log(" Extension installed");
});

// OPEN SIDE PANEL ON CLICK
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});
