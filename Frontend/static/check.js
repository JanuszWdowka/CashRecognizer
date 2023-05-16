let fileId = "id_banknoteImage";
let deleteId = "delete";
let labelTag = "label";

let banknotePhotoLabel = "Banknote Photo";

let check = document.getElementById(fileId);
let del = document.getElementById(deleteId);
let labels = document.getElementsByTagName(labelTag);

check.onchange = function () {
    setLabel(fileId, banknotePhotoLabel, check);
};

del.onclick = function () {
    handleDelete(fileId, banknotePhotoLabel, check);
}