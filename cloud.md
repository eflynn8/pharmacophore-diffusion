# Deploy PharmacoForge on Google Cloud



Run the following code in the terminal.  You will have to select a project.  The webapp will be running at the service URL provided.

## 1. Select Project
<walkthrough-project-setup billing="true"></walkthrough-project-setup>

## 2. Run

```html
<walkthrough-code-snippet>
<pre>
gcloud config set project {{project-id}} && bash deploy.sh
</pre>
</walkthrough-code-snippet>
```

```bash
bash deploy.sh
```


