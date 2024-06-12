var language;
var inp_dict = {};
var count = 1;
var res, drug_result;

const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add(className, "chat");
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

    // for adding avatars
    // let robots = document.getElementsByClassName("robot");
    // let users = document.getElementsByClassName("user");
    // for (let i = 0; i < robots.length; i++) {
    //     robots[i].innerHTML = '🤖';
    //     users[i].innerHTML = '👤';
    // }
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

    fetch("/func", {
        method: "POST",
        headers: {
            "Content-Type": "application/json;charset=UTF-8"
        },
        body: data_input
    })
    .then(response => {
        if (response.status === 200) {
            return response.json();
        } else {
            throw new Error("Request failed");
        }
    })
    .then(msg => {
        if (msg === 'Not enough information') {
            if (language === 'en') {
                incomingChatLi.innerHTML = `<p>Thank you for the description. To assist you more accurately, I need additional information. Please describe your symptoms in detail, including:<br>
                • Duration<br>
                • Severity<br>
                • Any other related symptoms
                </p>`;
            } else {
                incomingChatLi.innerHTML = `<p>謝謝您的描述。為了更準確地幫助您，我需要更多的資訊。請詳細描述您的症狀，包括：<br>
                - 持續的時間 <br>
                - 嚴重程度 <br>
                - 是否有其他相關症狀
                </p>`;
            }
        } else {
            if (language === 'en') {
                incomingChatLi.innerHTML = `<p>The possible disease you have is <strong>${msg}</strong></p>`;
                var drugChatLi = createChatLi("To further confirm, you may want to consider seeing a doctor or undergoing additional tests. Would you like to know about over-the-counter medications related to this condition or symptoms?", "chat-incoming");
            } else {
                incomingChatLi.innerHTML = `<p>您可能患有的疾病是<strong>${msg}</strong></p>`;
                var drugChatLi = createChatLi("為了進一步確認，您可以考慮去看醫生或進一步檢查。請問您是否想要了解與此疾病或症狀相關的成藥資訊?", "chat-incoming");
            }
            res = preSendDrug(input, language);
            chatbox.insertBefore(drugChatLi, opt_elmt);
            opt_elmt.style.display = "block";
        }
        chatbox.scrollTo(0, chatbox.scrollHeight);
    })
    .catch(error => {
        console.error(error);
    });
}

async function preSendDrug(input, lang) {
    const option = 'yes';
    const data = {
        option: option,
        user_input: input,
        lang: lang
    };

    return fetch("/drugs", {
        method: "POST",
        headers: {
            "Content-Type": "application/json;charset=UTF-8"
        },
        body: JSON.stringify(data)
    })
}

document.addEventListener('DOMContentLoaded', function() {
    const optNoEn = document.getElementById("opt_no-en");
    const optNoZh = document.getElementById("opt_no-zh");
    const optYesEn = document.getElementById("opt_yes-en");
    const optYesZh = document.getElementById("opt_yes-zh");
    const sendBtnEn = document.getElementById("sendBTN-en");
    const sendBtnZh = document.getElementById("sendBTN-zh");
    
    if (optNoEn) {
        optNoEn.addEventListener("click", function() {
            handleOptionNo("en");
        });
    }
    if (optNoZh) {
        optNoZh.addEventListener("click", function() {
            handleOptionNo("zh");
        });
    }
    if (optYesEn) {
        optYesEn.addEventListener("click", function() {
            handleOptionYes("en");
        });
    }
    if (optYesZh) {
        optYesZh.addEventListener("click", function() {
            handleOptionYes("zh");
        });
    }
    
    function handleOptionNo(lang) {
        let optElmt, chatbox, drg, cpt, drugTb;
        
        if (lang === 'en') {
            optElmt = document.getElementById("opt-en");
            chatbox = document.querySelector(".chatbox-en");
            drg = document.getElementById("drugs-en");
            cpt = document.getElementById("cpt_drugs-en");
            drugTb = document.getElementById("tb_drugs-en");
        } else {
            optElmt = document.getElementById("opt-zh");
            chatbox = document.querySelector(".chatbox-zh");
            drg = document.getElementById("drugs-zh");
            cpt = document.getElementById("cpt_drugs-zh");
            drugTb = document.getElementById("tb_drugs-zh");
        }
        
        optElmt.style.display = "none";
        chatbox.insertBefore(createChatLi("No.", "chat-outgoing"), optElmt);
        sendBtnEn.style.backgroundColor = "red";
        sendBtnZh.style.backgroundColor = "red";
        
        if (lang === 'en') {
            sendBtnEn.innerHTML = "Restart";
        } else {
            sendBtnZh.innerHTML = "重新開始";
        }
        
        sendBtnEn.onclick = reset;
        sendBtnZh.onclick = reset;
        chatbox.scrollTo(0, chatbox.scrollHeight);
    }
    
    function handleOptionYes(lang) {
        let optElmt, chatbox, drg, cpt, drugTb;
        
        if (lang === 'en') {
            optElmt = document.getElementById("opt-en");
            chatbox = document.querySelector(".chatbox-en");
            drg = document.getElementById("drugs-en");
            cpt = document.getElementById("cpt_drugs-en");
            drugTb = document.getElementById("tb_drugs-en");
        } else {
            optElmt = document.getElementById("opt-zh");
            chatbox = document.querySelector(".chatbox-zh");
            drg = document.getElementById("drugs-zh");
            cpt = document.getElementById("cpt_drugs-zh");
            drugTb = document.getElementById("tb_drugs-zh");
        }
        
        optElmt.style.display = "none";
        chatbox.insertBefore(createChatLi("Yes.", "chat-outgoing"), optElmt);
        const incomingChatLi = createChatLi("Thinking...", "chat-incoming");
        chatbox.insertBefore(incomingChatLi, optElmt);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        
        // check if the fetch is done
        res.then((obj) => obj.json()).then((json) => {
            drug_result = json;
            msg = drug_result;
            if (Object.keys(msg).length === 0) {
                if (lang === 'en') {
                    incomingChatLi.innerHTML = `<p>Sorry, no over-the-counter drug recommendations found.</p>`;
                } else {
                    incomingChatLi.innerHTML = `<p>抱歉，於我們的資料中，找不到適合的成藥推薦。</p>`;
                }
            } else {
                incomingChatLi.remove();
                drg.style.display = "block";
                if (lang === 'en') {
                    cpt.innerHTML = `Top ${Object.keys(msg).length} over-the-counter drug recommendations:`;
                    chatbox.appendChild(createChatLi("The table above is the over-the-counter drug recommendations for your symptoms.<br> However, everyone's situation is different, so be sure to consult your doctor or pharmacist before using any medication.", "chat-incoming"));
                    chatbox.appendChild(createChatLi("As a reminder, always consult a pharmacist's advice again before purchasing or using any medication to ensure safe usage. Additionally, if your symptoms persist or worsen, seek medical attention promptly. Wishing you a speedy recovery!", "chat-incoming"));
                }
                else {
                    cpt.innerHTML = `成藥推薦前 ${Object.keys(msg).length} 名:`;
                    chatbox.appendChild(createChatLi("上面的表格為針對您症狀的成藥建議。<br>不過，每個人的情況不同，請務必在使用前諮詢您的醫生或藥師。", "chat-incoming"));
                    chatbox.appendChild(createChatLi("提醒您，在購買或使用任何藥物前，務必再次詢問藥師的建議以確保用藥安全。同時，如果症狀持續或加重，請及時就醫。祝您早日康復！", "chat-incoming"));
                }
                for (let key in msg) {
                    var row = drugTb.insertRow();
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
            sendBtnEn.style.backgroundColor = "red";
            sendBtnZh.style.backgroundColor = "red";

            if (lang === 'en') {
                sendBtnEn.innerHTML = "Restart";
            } else {
                sendBtnZh.innerHTML = "重新開始";
            }

            sendBtnEn.onclick = reset;
            sendBtnZh.onclick = reset;
        });
    }
});
