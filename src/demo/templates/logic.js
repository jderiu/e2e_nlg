var theFrame = document.getElementById('iframe');
var theWin = theFrame.contentWindow;
var theDoc = theFrame.contentDocument || theFrame.contentWindow.document;
var json = theDoc;
var msgs = JSON.parse(theDoc);

for (var i = 0, l = msgs.length; i < l; i++) {
    var msg = msgs[i];
    var div = document.createElement('div');
    div.innerHTML = 'Hello ' + msg.user + ' your Id is: ' + msg.ID + 'and your message is: ' + msg.message + ' it has ' + msg.replies + ' replies';
    document.body.appendChild(div);
}