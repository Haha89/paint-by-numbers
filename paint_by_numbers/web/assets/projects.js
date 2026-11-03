// Saved projects: the picture and the settings, kept in the localStorage of the browser.
// The index lists the projects with their thumbnails, each project is under its own key with its picture, so that
// listing them doesn't read every picture.

(function () {
    "use strict";

    const INDEX = "paint-by-numbers:projects";
    const PREFIX = "paint-by-numbers:project:";
    const FULL = "The browser has no room left for this project: delete a saved project and try again.";

    function readIndex() {
        try {
            const index = JSON.parse(localStorage.getItem(INDEX));
            return Array.isArray(index) ? index : [];
        } catch (problem) {
            return [];
        }
    }

    // The id of the clicked button of a list, null when the list was only drawn again
    function clicked() {
        const triggered = dash_clientside.callback_context.triggered;
        if (!triggered.length || !triggered[0].value) return null;
        return JSON.parse(triggered[0].prop_id.split(".")[0]).index;
    }

    // Store a project built by the server, or only list the saved ones when the page loads
    function save(project) {
        const index = readIndex();
        if (!project) return [index, "", "dimmed"];
        const entry = {id: project.id, name: project.name, thumbnail: project.thumbnail, saved_at: project.saved_at};
        try {
            localStorage.setItem(PREFIX + project.id, JSON.stringify(project));
            localStorage.setItem(INDEX, JSON.stringify([entry, ...index.filter((other) => other.id !== project.id)]));
        } catch (problem) {
            // A failed write leaves the previous version of a project, but a new one must not stay unlisted
            if (!index.some((other) => other.id === project.id)) localStorage.removeItem(PREFIX + project.id);
            console.warn("The project can't be saved", problem);
            return [index, FULL, "red"];
        }
        return [readIndex(), "Project saved in this browser.", "green"];
    }

    function open(_clicks) {
        const id = clicked();
        if (id === null) return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
        try {
            const project = JSON.parse(localStorage.getItem(PREFIX + id));
            if (project) return [project, "", "dimmed"];
        } catch (problem) {
            console.warn("The project can't be read", problem);
        }
        return [dash_clientside.no_update, "This project can't be read anymore.", "red"];
    }

    function remove(_clicks) {
        const id = clicked();
        if (id === null) return dash_clientside.no_update;
        localStorage.removeItem(PREFIX + id);
        const index = readIndex().filter((project) => project.id !== id);
        localStorage.setItem(INDEX, JSON.stringify(index));
        return index;
    }

    window.dash_clientside = Object.assign({}, window.dash_clientside, {projects: {save, open, remove}});
})();
