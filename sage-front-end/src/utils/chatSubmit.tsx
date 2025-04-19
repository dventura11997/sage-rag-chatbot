// FormSubmit.ts
interface FormSubmitResponse {
    success: boolean;
    result?: any; // You can define a more specific type based on your API response
    error?: string;
  }

  async function chatSubmit(submitUrl: string, data: Record<string, any>): Promise<FormSubmitResponse> {
    const headers = {
      'Content-Type': 'application/json',
    };
  
    try {
      const response = await fetch(submitUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data),
      });
  
      const result = await response.json();
  
      if (response.ok) {
        return { success: true, result };
      } else {
        return { success: false, error: result.message || 'Form submission failed' };
      }
    } catch (error) {
      console.error(error);
      return { success: false, error: (error as Error).message || 'Form submission failed' };
    }
  }
  
  export default chatSubmit;