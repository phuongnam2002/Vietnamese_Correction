const enter_btn = document.getElementById("enter-btn")
const desc_textarea = document.getElementById('desc-textarea')
const csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0]
const spiner = document.getElementById('spiner')
const output = document.getElementById("output")

const server_url = `${window.location.protocol}//${window.location.host}`

enter_btn.onclick = async function () {
    desciption = desc_textarea.value
    token = csrf_token.value
    spiner.style.display = "flex"

    await $.ajax({
        type: "POST",
        url: server_url,
        headers: {
            "X-CSRFToken": token
        },
        data: {
            "desciption": desciption,
        },
        success: function (result) {
            output.value= result['answer']
        },
        dataType: "json"
    });
    spiner.style.display = "none"
    output.style.display = "flex"
}