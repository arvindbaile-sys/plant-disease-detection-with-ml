document.getElementById('upload').onclick = async () => {
const file = document.getElementById('file').files[0];
if(!file){ alert('Select an image'); return; }
const fd = new FormData();
fd.append('file', file);
try{
const res = await fetch('http://localhost:5000/predict', {method:'POST', body: fd});
const data = await res.json();
document.getElementById('result').innerText = JSON.stringify(data, null, 2);
}catc