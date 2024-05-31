var language;
var inp_dict = {};
var count = 1;

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

    inp_dict[count] = input;
    count += 1;

    chatbox.insertBefore(createChatLi(input, "chat-outgoing"), opt_elmt);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    inp.value = "";

    const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
    chatbox.insertBefore(incomingChatLi, opt_elmt);
    chatbox.scrollTo(0, chatbox.scrollHeight);
    data_input = JSON.stringify(inp_dict);

    $.ajax({
        method: "POST",
        url: "/func",
        dataType: 'json',
        contentType: "application/json; charset=utf-8",
        data: data_input
    })
    .done(function( msg ) {
        if (msg == 'Not enough information') {
            if (language == 'en') {
                incomingChatLi.innerHTML = `<p>The information you provide is not enough for disease prediction. Please elaborate more.</p>`;
            }
            else {
                incomingChatLi.innerHTML = `<p>您提供的資訊不足以進行疾病判別，請再詳述您的狀況。</p>`;
            }
        }
        else {
            if (language == 'en') {
                incomingChatLi.innerHTML = `<p>The possible disease you have is 
                    <strong>${msg}</strong></p>`;
                var drugChatLi = 
                    createChatLi("Do you want to get information about disease-related over-the-counter drugs?", "chat-incoming");
            }
            else {
                incomingChatLi.innerHTML = `<p>您可能患有的疾病是
                    <strong>${msg}</strong></p>`;
                var drugChatLi = 
                    createChatLi("您是否想獲取與疾病、症狀相關的成藥資訊?", "chat-incoming");
            }
            chatbox.insertBefore(drugChatLi, opt_elmt);
            opt_elmt.style.display = "block";
        }
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
}

$(document).ready(function(){
    $('#opt_no-en, #opt_no-zh').click(function(){
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
        opt_elmt.style.display = "none";
        chatbox.insertBefore(createChatLi("No.", "chat-outgoing"), opt_elmt);

        sdBTN.style.backgroundColor = "red";
        if (language == 'en') sdBTN.innerHTML = "Restart";
        else sdBTN.innerHTML = "重新開始";
        sdBTN.onclick = reset;
        chatbox.scrollTo(0, chatbox.scrollHeight);
    });
    $('#opt_yes-en, #opt_yes-zh').click(function(){
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
            data: { option: option, user_input: input, lang: language}
        })
        .done(function( msg ) {
            // msg is a json with key and value
            if (Object.keys(msg).length == 0) {
                if (language == 'en') {
                    incomingChatLi.innerHTML = `<p>Sorry, no over-the-counter drug recommendations found.</p>`;
                }
                else {
                    incomingChatLi.innerHTML = `<p>抱歉，於我們的資料中，找不到適合的成藥推薦。</p>`;
                }
            }    
            else{
                incomingChatLi.remove()
                drg.style.display = "block";
                if (language == 'en') {
                    cpt.innerHTML = `Top ${Object.keys(msg).length} over-the-counter drug recommendations:`;
                    incomingChatLi.innerHTML += `<p>Please ask the pharmacist again when going to the pharmacy.</p>`;
                }
                else {
                    cpt.innerHTML = `成藥推薦前 ${Object.keys(msg).length} 名:`;
                    incomingChatLi.innerHTML += `<p>請在去藥局賣藥時，再次詢問藥師。</p>`;
                }
                for (var key in msg) {
                    var row = drug_tb.insertRow();
                    var rank = row.insertCell(0);
                    var name = row.insertCell(1);
                    var simi = row.insertCell(2);
                    var idc = row.insertCell(3);

                    info = msg[key].split('@');

                    rank.innerHTML = key;
                    name.innerHTML = info[0];
                    simi.innerHTML = info[1];
                    idc.innerHTML = info[2];
                }
            }
            chatbox.scrollTo(0, chatbox.scrollHeight);
            sdBTN.style.backgroundColor = "red";
            if (language == 'en') sdBTN.innerHTML = "Restart";
            else sdBTN.innerHTML = "重新開始";
            sdBTN.onclick = reset;
        });
    });
});
