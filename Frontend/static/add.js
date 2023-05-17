/**
 * Plik funkcyjny dla addBanknote.html
 */
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

/**
 * Funkcja ustawiająca etykietę dla zdjęcia przodu banknotu
 */
front.onchange = function () {
    setLabel(fileIdFront, banknoteFrontLabel, front);
};

/**
 * Funkcja ustawiająca etykietę dla zdjęcia tyłu banknotu
 */
back.onchange = function () {
    setLabel(fileIdBack, banknoteBackLabel, back);
};

/**
 * Funkcja czyszcząca pola zdjęć w formularzu
 */
del.onclick = function () {
    handleDelete(fileIdFront, banknoteFrontLabel, front)
    handleDelete(fileIdBack, banknoteBackLabel, back)
}