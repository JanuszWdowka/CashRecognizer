let fileIdFront = "id_banknoteFront";
let fileIdBack = "id_banknoteBack";
let deleteId = "delete";
let labelTag = "label";

let banknoteFrontLabel = "BanknoteFront";
let banknoteBackLabel = "BanknoteBack";


let front = document.getElementById(fileIdFront);
let back = document.getElementById(fileIdBack);
let del = document.getElementById(deleteId);
let labels = document.getElementsByTagName(labelTag);

front.onchange = function () {
    setLabel(fileIdFront, banknoteFrontLabel, front);
};

back.onchange = function () {
    setLabel(fileIdBack, banknoteBackLabel, back);
};

del.onclick = function () {
    handleDelete(fileIdFront, banknoteFrontLabel, front)
    handleDelete(fileIdBack, banknoteBackLabel, back)
}