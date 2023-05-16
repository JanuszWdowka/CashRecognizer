function getLabelElement(id) {
    let label;

    for( var i = 0; i < labels.length; i++ ) {
        if (labels[i].htmlFor == id)
            label = labels[i];
    }

    return label;
}

function setLabel(id, buttonLabel, file) {
    let label = getLabelElement(id);

    if(file.value == '') {
        label.innerHTML = buttonLabel;
    }
    else {
        label.innerHTML = document.getElementById(id).value.slice(12,1000);
    }
}

function handleDelete(id, buttonLabel, file) {
    let label = getLabelElement(id);

    file.value = '';
    label.innerHTML = buttonLabel;
}