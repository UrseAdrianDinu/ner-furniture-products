import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import {FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators} from "@angular/forms";
import {HttpClient, HttpClientModule, HttpHeaders} from "@angular/common/http";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, ReactiveFormsModule, HttpClientModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'veri';
  inputText: string = '';
  modelUrl = "http://localhost:5000/predict";

  responseData: Array<any> | null = null;

  inputForm = new FormGroup({
    inputText: new FormControl('', {nonNullable:true, validators:[Validators.required]})
  });

  modelOutputText: string[] = [];

  constructor(protected http: HttpClient) {
  }

  predict() {
    var urlInputUser = this.inputForm.getRawValue().inputText
    const body = { url: urlInputUser};
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });
    this.http.post(this.modelUrl, body, { headers, observe: 'response' })
      .subscribe(response => {
          const responseBody = response.body as { products: string[] };
          this.modelOutputText = responseBody.products;  // Store the products array
        },
        error => {
          console.error('Error:', error);
        });
  }
}
