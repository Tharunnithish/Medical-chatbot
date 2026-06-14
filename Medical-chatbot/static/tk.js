
chatBtn.addEventListener('click', () => {
  chatBox.style.display = 'flex';
  chatBtn.style.display = 'none';
});
closeBtn.addEventListener('click', () => {
  chatBox.style.display = 'none';
  chatBtn.style.display = 'block';
});
sendBtn.addEventListener('click', () => {
  const text = userInput.value.trim();
  if (!text) return;
  
  const userMessage = document.createElement('div');
  userMessage.classList.add('message', 'user');
  userMessage.textContent = text;
  chatBody.appendChild(userMessage);
  
  userInput.value = '';
  
  const botMessage = document.createElement('div');
  botMessage.classList.add('message', 'bot');
  botMessage.textContent = "Got it! We'll get back to you soon.";
  chatBody.appendChild(botMessage);
  
  chatBody.scrollTop = chatBody.scrollHeight;
});