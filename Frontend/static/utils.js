/**
 * Plik z funkcjami narzędziowymi używanymi przez inne pliki JS i HTMLowe projektu
 */

/**
 * Funkcja pobierająca etykietę elemementu
 * @param id id elementu HTMLowego
 * @returns {*} treść etykiety
 */
function getLabelElement(id) {
    let label;

    for( var i = 0; i < labels.length; i++ ) {
        if (labels[i].htmlFor == id)
            label = labels[i];
    }

    return label;
}

/**
 * Funkcja ustawiająca etykietę
 * @param id id elementu HTMLowego
 * @param buttonLabel nazwa przycisku jaką chcemy ustawić
 * @param file element HTMLowy posiadający plik
 */
function setLabel(id, buttonLabel, file) {
    let label = getLabelElement(id);

    if(file.value == '') {
        label.innerHTML = buttonLabel;
    }
    else {
        label.innerHTML = document.getElementById(id).value.slice(12,1000);
    }
}

/**
 * Funkcja czyszcząca dane po pliku dodanym do formularza
 * @param id id elementu HTMLowego
 * @param buttonLabel nazwa przycisku jaką chcemy ustawić
 * @param file element HTMLowy posiadający plik
 */
function handleDelete(id, buttonLabel, file) {
    let label = getLabelElement(id);

    file.value = '';
    label.innerHTML = buttonLabel;
}