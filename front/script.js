function getInput() {
  const inputElt = document.getElementById("sentence-input");
  const value = inputElt.value;
  inputElt.value = "";
  return value;
}

async function runClassification(input) {
  const response = await fetch('http://localhost:5000/execute', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: input
    })
  });

  if (!response.ok) {
    throw new Error("Error with api call");
  }

  const data = await response.json();
  return data.result;
}

async function run() {
   const input = getInput();

  if (input !== "") {
    document.getElementById("input-echo").innerHTML = `"${input}"`
    const resultElt = document.getElementById("result")
    resultElt.classList.remove("positive")
    resultElt.classList.remove("negative")
    resultElt.innerHTML = "Your sentence is ...";
    const result = await runClassification(input);
    const negative = result[0];
    const positive = result[4];
    if (positive > negative) {
      resultElt.innerHTML = `Your sentence is ${Math.round(positive * 100 * 10) / 10}% likely to be POSITIVE`
      resultElt.classList.add("positive")
    } else {
      resultElt.innerHTML = `Your sentence is ${Math.round(negative * 100 * 10) / 10}% likely to be NEGATIVE`
      resultElt.classList.add("negative")
    }
  }
}

document.getElementById("submit-button").addEventListener("click", run);
document.getElementById("sentence-input").addEventListener("keydown", async function(event) {
  if (event.key === "Enter") {
    event.preventDefault()
    await run()
  }
});
