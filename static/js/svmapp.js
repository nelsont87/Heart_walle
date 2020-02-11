function animation() {
    d3.json(`/svmstat`).then(function(data) {
        var fix = 0;
        if (data["improvement"] != 0 ){
            fix = data["improvement"];
        };

        var canvasR = document.getElementById("result");
        if (data["prediction"][0] == 0){
            canvasR.textContent +=  "You are Healthy!"
            gsap.to("#healthy", {autoAlpha:1, visibility: "visible", duration:1});
        } else if (data["prediction"][0] == 1){
            canvasR.textContent +=  "Consult your Doctor, there may be some issues."
            gsap.to("#doctor", {autoAlpha:1, visibility: "visible", duration:1});
        };

        if (fix != 0){
            var ctx = document.getElementById("fix");

            if (fix == "cholesterol"){
                ctx.textContent += "Be careful of cholesterol intake.";
                gsap.to("#cholC", {autoAlpha:1, visibility: "visible", duration:1});
            } 
            else if (fix == "weight"){
                ctx.textContent += "Motivate yourself to shed some unnecessary weight.";
                gsap.to("#weightC", {autoAlpha:1, visibility: "visible", duration:1});
            }
            else if (fix == "alcohol"){
                ctx.textContent += "Lessen your intake of alcohol.";
                gsap.to("#alcoC", {autoAlpha:1, visibility: "visible", duration:1});
            }
            else if (fix == "BP"){
                ctx.textContent += "Monitor your blood pressure closely.";
                gsap.to("#BP", {autoAlpha:1, visibility: "visible", duration:1});
            }
            else if (fix = "smoking"){
                ctx.textContent += "Time to help others and yourself by not smoking.";
                gsap.to("#smokeC", {autoAlpha:1, visibility: "visible", duration:1});
            }
            else if (fix == "glucose"){
                ctx.textContent += "Be careful of sugar intake.";
                gsap.to("#glucC", {autoAlpha:1, visibility: "visible", duration:1});
            }
            else if (fix == "active"){
                ctx.textContent += "Motivate yourself to move.";
                gsap.to("#activeC", {autoAlpha:1, visibility: "visible", duration:1});
            }
        };
        
    });
};
animation();
