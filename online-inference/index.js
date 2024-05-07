window.onload = function() {
    let parentDiv = document.getElementById("grid");

    for (let i = 0; i < 28; i++){
        for (let j = 0; j < 28; j++){
            const div = document.createElement("div");
            div.classList.add("cell");
            parentDiv.appendChild(div);
        }
    }
}