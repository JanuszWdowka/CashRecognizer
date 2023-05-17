/**
 * Plik funkcyjny dla checkBanknote.html
 */
let fileId = "id_banknoteImage";
let deleteId = "delete";
let labelTag = "label";

let banknotePhotoLabel = "Banknote Photo";

let check = document.getElementById(fileId);
let del = document.getElementById(deleteId);
let labels = document.getElementsByTagName(labelTag);

/**
 * Funkcja ustawiająca etykietę dla zdjęcia banknotu dodanego przez użytkownika
 */
check.onchange = function () {
    setLabel(fileId, banknotePhotoLabel, check);
};

/**
 * Funkcja czyszcząca pole zdjęcie w formularzu
 */
del.onclick = function () {
    handleDelete(fileId, banknotePhotoLabel, check);
}