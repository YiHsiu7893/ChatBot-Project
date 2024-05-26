var language;

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent = 
      className === "chat-outgoing" ? `<p>${message}</p>` : `<p>${message}</p>`;
    chatLi.innerHTML = chatContent;
    return chatLi;
};

function closeChat() {
    let ui_en = document.querySelector(".chatBot-en");
    let ui_zh = document.querySelector(".chatBot-zh");
    if (ui_en.style.display != 'none') {
        ui_en.style.display = "none";
        document.querySelector(".end-en").style.display = "block";
    }
    else {
        ui_zh.style.display = "none";
        document.querySelector(".end-zh").style.display = "block";
    }
}

function reset() {
    // GET /main to restart the chat
    window.location.href = "/main";
}

document.addEventListener('DOMContentLoaded', function() {
    inp_en = document.getElementById("user_input-en");
    inp_zh = document.getElementById("user_input-zh");
    if (inp_en) {
        inp_en.addEventListener("keypress", e => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMsg();
            }
        });
    }
    if (inp_zh) {
        inp_zh.addEventListener("keypress", e => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMsg();
            }
        });
    }
});

function changeLang(lang) {
    document.getElementById('lang').style.display = 'none';
    if (lang == 'en') {
        language = 'en';
        document.getElementsByClassName('chatBot-zh')[0].style.display = 'none';
        document.getElementsByClassName('chatBot-en')[0].style.display = 'block';
    }
    else {
        language = 'zh';
        document.getElementsByClassName('chatBot-en')[0].style.display = 'none';
        document.getElementsByClassName('chatBot-zh')[0].style.display = 'block';
    }
}

function sendMsg() {
    if (language == 'en') {
        inp = document.getElementById("user_input-en");
        opt_elmt = document.getElementById("opt-en");
        chatbox = document.querySelector(".chatbox-en");
    }
    else {
        inp = document.getElementById("user_input-zh");
        opt_elmt = document.getElementById("opt-zh");
        chatbox = document.querySelector(".chatbox-zh");
    }

    const input = inp.value;
    if (!input) return;

    chatbox.insertBefore(createChatLi(input, "chat-outgoing"), opt_elmt);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    inp.value = "";

    const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
    chatbox.insertBefore(incomingChatLi, opt_elmt);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    $.ajax({
        method: "POST",
        url: "/func",
        data: { user_input: input }
    })
    .done(function( msg ) {
        incomingChatLi.innerHTML = `<p>The possible disease you have is 
            <strong>${msg}</strong></p>`;
        var drugChatLi = 
            createChatLi("Do you want to get information about disease-related over-the-counter drugs?", "chat-incoming");
        chatbox.insertBefore(drugChatLi, opt_elmt);
        opt_elmt.style.display = "block";
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
}

$(document).ready(function(){
    if (language == 'en') {
        sdBTN = document.getElementById("sendBTN-en");
        opt_elmt = document.getElementById("opt-en");
        chatbox = document.querySelector(".chatbox-en");
        drg = document.getElementById("drugs-en");
        cpt = document.getElementById("cpt_drugs-en");
        drug_tb = document.getElementById("tb_drugs-en");
        // opt_yes = document.querySelector("#opt-en #opt_yes");
        // opt_no = document.querySelector("#opt-en #opt_no");
    }
    else {
        sdBTN = document.getElementById("sendBTN-zh");
        opt_elmt = document.getElementById("opt-zh");
        chatbox = document.querySelector(".chatbox-zh");
        drg = document.getElementById("drugs-zh");
        cpt = document.getElementById("cpt_drugs-zh");
        drug_tb = document.getElementById("tb_drugs-zh");
        // opt_yes = document.querySelector("#opt-zh #opt_yes");
        // opt_no = document.querySelector("#opt-zh #opt_no");
    }

    $('#opt_no-en, #opt_no-zh').click(function(){
        opt_elmt.style.display = "none";
        chatbox.insertBefore(createChatLi("No.", "chat-outgoing"), opt_elmt);

        sdBTN.style.backgroundColor = "red";
        sdBTN.innerHTML = "Restart";
        sdBTN.onclick = reset;
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
    $('#opt_yes-en, #opt_yes-zh').click(function(){
        opt_elmt.style.display = "none";
        var option = 'yes';
        chatbox.insertBefore(createChatLi("Yes.", "chat-outgoing"), opt_elmt);
        outgo_lis = document.querySelectorAll('.chat-outgoing');
        var input = outgo_lis[outgo_lis.length - 2].innerText;
        const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
        chatbox.insertBefore(incomingChatLi, opt_elmt);
        chatbox.scrollTo(0, chatbox.scrollHeight);
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
                drg.style.display = "block";
                if (language == 'en') {
                    cpt.innerHTML = `Top ${Object.keys(msg).length} over-the-counter drug recommendations:`;
                }
                else {
                    cpt.innerHTML = `成藥推薦前 ${Object.keys(msg).length} 名:`;
                }
                for (var key in msg) {
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
            sdBTN.style.backgroundColor = "red";
            sdBTN.innerHTML = "Restart";
            sdBTN.onclick = reset;
        });
    });
});
