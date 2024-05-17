const opt_elmt = document.getElementById("opt");
const chatbox = document.querySelector(".chatbox");

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = 
      className === "chat-outgoing" ? `<p>${message}</p>` : `<p>${message}</p>`;
    chatLi.innerHTML = chatContent;
    return chatLi;
};

function closeChat() {
    let ui = document.querySelector(".chatBot");
    if (ui.style.display != 'none') {
        ui.style.display = "none";
        document.querySelector(".end").style.display = "block";
    }
}

function reset() {
    // GET /main to restart the chat
    window.location.href = "/main";
}

$(document).ready(function(){
    $('#sendBTN').click(function(){
        const chatbox = document.querySelector(".chatbox");
        const opt_elmt = document.getElementById("opt");
        var inp = document.getElementById("user_input");
        var input = inp.value
        if (!input) return;

        chatbox.insertBefore(createChatLi(input, "chat-outgoing"), opt_elmt);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        inp.value = "";
        // $('#user_input').val("");

        const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
        chatbox.insertBefore(incomingChatLi, opt_elmt);
        chatbox.scrollTo(0, chatbox.scrollHeight);

        $.ajax({
            method: "POST",
            url: "/func",
            data: { user_input: input}
        })
        .done(function( msg ) {
            incomingChatLi.innerHTML = `<p>The possible disease you have is 
                <strong>${msg}</strong></p>`;
            var drugChatLi = 
                createChatLi("Do you want to get information about disease-related over-the-counter drugs?", "chat-incoming");
            chatbox.insertBefore(drugChatLi, opt_elmt);
            chatbox.scrollTo(0, chatbox.scrollHeight);
            $('#opt').show();
        });
    });

    $('#opt_no').click(function(){
        $('#opt').hide();
        const sdBTN = document.getElementById("sendBTN");
        sdBTN.style.backgroundColor = "red";
        sdBTN.innerHTML = "Restart";
        sdBTN.onclick = reset;
    });
    $('#opt_yes').click(function(){
        $('#opt').hide();
        var option = 'yes';
        const chatbox = document.querySelector(".chatbox");
        outgo_lis = document.querySelectorAll('.chat-outgoing');
        var input = outgo_lis[outgo_lis.length - 1].innerText;
        const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
        chatbox.insertBefore(incomingChatLi, opt_elmt);
        $.ajax({
            method: "POST",
            url: "/drugs",
            data: { option: option, user_input: input}
        })
        .done(function( msg ) {
            // msg is a json with key and value
            if (Object.keys(msg).length == 0) {
                incomingChatLi.innerHTML = `<p>Sorry, no over-the-counter drug recommendations found.</p>`;
            }
            else{
                incomingChatLi.remove()
                $('#drugs').show();
                const cpt = document.getElementById("cpt_drugs");
                cpt.innerHTML = `Top ${Object.keys(msg).length} over-the-counter drug recommendations:`;
                for (var key in msg) {
                    const drug_tb = document.getElementById("tb_drugs");
                    var row = drug_tb.insertRow();
                    var rank = row.insertCell(0);
                    var name = row.insertCell(1);
                    var simi = row.insertCell(2);
                    var idc = row.insertCell(3);

                    info = msg[key].split(',');

                    rank.innerHTML = key;
                    name.innerHTML = info[0];
                    simi.innerHTML = info[1];
                    idc.innerHTML = info[2];
                }
                incomingChatLi.innerHTML += `<p>Please ask the pharmacist again when going to the pharmacy.</p>`;
            }
            chatbox.scrollTo(0, chatbox.scrollHeight);
            const sdBTN = document.getElementById("sendBTN");
            sdBTN.style.backgroundColor = "red";
            sdBTN.innerHTML = "Restart";
            sdBTN.onclick = reset;
        });
    });
});
